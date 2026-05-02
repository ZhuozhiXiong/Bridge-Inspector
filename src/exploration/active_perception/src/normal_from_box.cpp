#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>          // 1. 补上
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Core>
#include <Eigen/Geometry>                    // 2. Quaternion 需要
#include <unordered_set>                     // 3. 哈希表
#include <vector>

class NormalFromBox {
public:
  NormalFromBox() : nh_("~"), acc_cloud_(new pcl::PointCloud<pcl::PointXYZ>) {
    /* 参数读取 */
    std::vector<double> smin, smax;
    nh_.param("search_min", smin, std::vector<double>{-1.0, -1.0, -1.0});
    nh_.param("search_max", smax, std::vector<double>{ 1.0,  1.0,  1.0});
    search_min_ << smin[0], smin[1], smin[2];
    search_max_ << smax[0], smax[1], smax[2];
    nh_.param("k_search_radius", kSearchRadius_, 0.5);
    nh_.param("voxel_size", voxel_, 0.05f);          // 4. 去重分辨率

    /* 人为 query 点 */
    query_pts_.emplace_back(0.0, 0.0, 0.0);
    query_pts_.emplace_back(0.5, 0.0, 0.2);

    /* 发布 / 订阅 */
    sub_   = nh_.subscribe<sensor_msgs::PointCloud2>("/cloud", 10,
                                                     &NormalFromBox::cloudCb, this);
    pub_crop_ = nh_.advertise<sensor_msgs::PointCloud2>("/cropped_cloud", 1);
    pub_pose_ = nh_.advertise<geometry_msgs::PoseArray>("/query_poses", 1);
    pub_map_  = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_map", 1);   // 5. 累积云
  }

  void cloudCb(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  /* ---- 4.1 解码 + 去重累积 ---- */
  pcl::PointCloud<pcl::PointXYZ>::Ptr raw(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*msg, *raw);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setLeafSize(voxel_, voxel_, voxel_);
  voxel_filter.setInputCloud(raw);
  voxel_filter.filter(*filtered);

  pcl::PointCloud<pcl::PointXYZ>::Ptr add(new pcl::PointCloud<pcl::PointXYZ>);
  add->reserve(filtered->size());
  const float inv = 1.0f / voxel_;
  for (const auto& p : filtered->points) {
    VoxelKey k{ static_cast<int64_t>(p.x * inv),
                static_cast<int64_t>(p.y * inv),
                static_cast<int64_t>(p.z * inv) };
    if (occupied_.insert(k).second) add->push_back(p);
  }
  *acc_cloud_ += *add;

  /* 发布累积地图 */
  // sensor_msgs::PointCloud2 map_msg;
  // pcl::toROSMsg(*acc_cloud_, map_msg);
  // map_msg.header       = msg->header;
  // map_msg.header.frame_id = "map";
  // pub_map_.publish(map_msg);

  /* ---- 4.2 CropBox ---- */
  pcl::CropBox<pcl::PointXYZ> crop;
  pcl::PointCloud<pcl::PointXYZ>::Ptr crop_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  crop.setInputCloud(acc_cloud_);
  crop.setMin(Eigen::Vector4f(search_min_.x(), search_min_.y(), search_min_.z(), 1.0));
  crop.setMax(Eigen::Vector4f(search_max_.x(), search_max_.y(), search_max_.z(), 1.0));
  crop.filter(*crop_cloud);
  if (crop_cloud->empty()) {
    ROS_WARN_THROTTLE(2.0, "crop cloud empty");
    return;
  }

  // 3. 构建 KD-Tree
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  kdtree->setInputCloud(crop_cloud);

  // 4. 估计每个 query 点的法向量
  // 混合策略：预计算整张法向量场 + 局部重建 fallback
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_w_normal(
      new pcl::PointCloud<pcl::PointNormal>);
  {
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> nest;
    nest.setInputCloud(crop_cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr nest_tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    nest.setSearchMethod(nest_tree);
    nest.setKSearch(20);          // 足够平滑，也可 setRadiusSearch
    pcl::PointCloud<pcl::Normal> normals;
    nest.compute(normals);
    pcl::concatenateFields(*crop_cloud, normals, *cloud_w_normal);
  }
  // 给预计算云建 kd-tree
  pcl::KdTreeFLANN<pcl::PointNormal>::Ptr normal_tree(
      new pcl::KdTreeFLANN<pcl::PointNormal>);
  normal_tree->setInputCloud(cloud_w_normal);

  geometry_msgs::PoseArray pose_arr;
  pose_arr.header.frame_id = msg->header.frame_id;
  pose_arr.header.stamp = ros::Time::now();

  const int min_neighbors = 10;   // 局部重建门槛
  for (const auto& q : query_pts_) {
    Eigen::Vector3d normal;
    bool use_local = false;

    // ------ 1. 尝试局部重建 ------
    // std::vector<int>   idx;
    // std::vector<float> dist2;
    // pcl::PointXYZ query;
    // query.x = q.x(); query.y = q.y(); query.z = q.z();
    // if (kdtree->radiusSearch(query, kSearchRadius_, idx, dist2) >= min_neighbors) {
    //   pcl::PointCloud<pcl::PointXYZ>::Ptr nn_cloud(
    //       new pcl::PointCloud<pcl::PointXYZ>);
    //   for (int i : idx) nn_cloud->push_back(crop_cloud->at(i));

    //   pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> nest_local;
    //   nest_local.setInputCloud(nn_cloud);
    //   pcl::search::KdTree<pcl::PointXYZ>::Ptr local_tree(
    //       new pcl::search::KdTree<pcl::PointXYZ>);
    //   nest_local.setSearchMethod(local_tree);
    //   nest_local.setKSearch(std::min<int>(20, nn_cloud->size()));
    //   pcl::PointCloud<pcl::Normal> local_normals;
    //   nest_local.compute(local_normals);

    //   normal << local_normals[0].normal_x,
    //             local_normals[0].normal_y,
    //             local_normals[0].normal_z;
    //   use_local = true;
    // }

    // ------ 2. 邻居不足 → 拿最近预计算法向量 ------
    if (!use_local) {
      pcl::PointNormal query_pn;
      query_pn.x = q.x(); query_pn.y = q.y(); query_pn.z = q.z();
      std::vector<int>   nn_idx(1);
      std::vector<float> nn_dist(1);
      if (normal_tree->nearestKSearch(query_pn, 1, nn_idx, nn_dist) == 0) {
        ROS_WARN_THROTTLE(2.0, "no normal for query (%.2f %.2f %.2f)", q.x(), q.y(), q.z());
        continue;
      }
      const pcl::PointNormal& np = cloud_w_normal->at(nn_idx[0]);
      normal << np.normal_x, np.normal_y, np.normal_z;
    }

    // ------ 3. 构造 Pose ------
    geometry_msgs::Pose p;
    p.position.x = q.x();
    p.position.y = q.y();
    p.position.z = q.z();

    Eigen::Vector3f z_axis = normal.normalized();
    Eigen::Vector3f x_axis = (std::abs(z_axis.dot(Eigen::Vector3f::UnitX())) < 0.9 ?
                              Eigen::Vector3f::UnitX() :
                              Eigen::Vector3f::UnitY()).cross(z_axis).normalized();
    Eigen::Vector3f y_axis = z_axis.cross(x_axis);
    Eigen::Matrix3f R;
    R.col(0) = x_axis; R.col(1) = y_axis; R.col(2) = z_axis;
    Eigen::Quaternionf qt(R);
    p.orientation.x = qt.x();
    p.orientation.y = qt.y();
    p.orientation.z = qt.z();
    p.orientation.w = qt.w();
    pose_arr.poses.push_back(p);
  }

  // 发布部分与你原代码相同
  // sensor_msgs::PointCloud2 crop_msg;
  // pcl::toROSMsg(*crop_cloud, crop_msg);
  // crop_msg.header = msg->header;
  // pub_crop_.publish(crop_msg);
  pub_pose_.publish(pose_arr);
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  ros::Publisher  pub_crop_, pub_pose_, pub_map_;   // 6. 统一名字
  pcl::PointCloud<pcl::PointXYZ>::Ptr acc_cloud_;
  Eigen::Vector3d search_min_, search_max_;
  double kSearchRadius_;
  std::vector<Eigen::Vector3d> query_pts_;
  float voxel_;                                     // 7. 去重分辨率

  /* 哈希去重结构 */
  struct VoxelKey {
    int64_t x{}, y{}, z{};
    bool operator==(const VoxelKey& o) const {
      return x == o.x && y == o.y && z == o.z;
    }
  };
  struct KeyHash {
    std::size_t operator()(const VoxelKey& k) const {
      return std::hash<int64_t>()(k.x) ^
             (std::hash<int64_t>()(k.y) << 1) ^
             (std::hash<int64_t>()(k.z) << 2);
    }
  };
  std::unordered_set<VoxelKey, KeyHash> occupied_;   // 8. 现在有了
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "normal_from_box_node");
  NormalFromBox nfb;
  ros::spin();
  return 0;
}