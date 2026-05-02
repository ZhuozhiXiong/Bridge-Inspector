#ifndef _MAP_ROS_H
#define _MAP_ROS_H

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>

#include <memory>
#include <random>

#include <mutex>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/filters/voxel_grid.h>
#include <ikd_Tree/ikd_Tree.h>

using std::shared_ptr;
using std::normal_distribution;
using std::default_random_engine;

namespace fast_planner {
class SDFMap;

class MapROS {
public:
  MapROS();
  ~MapROS();

  enum { POSE_STAMPED = 1, ODOMETRY = 2, INVALID_IDX = -10000 };

  void setMap(SDFMap* map);
  void init();

private:
  void depthPoseCallback(const sensor_msgs::ImageConstPtr& img,
                         const geometry_msgs::PoseStampedConstPtr& pose);
  void cloudPoseCallback(const sensor_msgs::PointCloud2ConstPtr& msg,
                         const geometry_msgs::PoseStampedConstPtr& pose);
  void extrinsicCallback(const nav_msgs::OdometryConstPtr &odom);
  void depthOdomCallback(const sensor_msgs::ImageConstPtr &img,
                         const nav_msgs::OdometryConstPtr &odom);
  void updateESDFCallback(const ros::TimerEvent& /*event*/);
  void visCallback(const ros::TimerEvent& /*event*/);
  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
  void ikdTreeTimerCallback(const ros::TimerEvent&);
  void globalCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

  void publishMapAll();
  void publishMapLocal();
  void publishESDF();
  void publishUpdateRange();
  void publishUnknown();
  void publishDepth();
  void publishMapFreeAll();
  void publishIkdTreeCloud(const ros::TimerEvent&);
  void publishMapSeen();
  void publishMapExpl();
  void publishMapCover();

  void processDepthImage();

  SDFMap* map_;
  KD_TREE<PointType> ikd_tree_;
  // may use ExactTime?
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped>
      SyncPolicyImagePose;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
      SyncPolicyImageOdom;
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> SynchronizerImagePose;
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImageOdom>> SynchronizerImageOdom;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
                                                          geometry_msgs::PoseStamped>
      SyncPolicyCloudPose;
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyCloudPose>> SynchronizerCloudPose;

  ros::NodeHandle node_;
  shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
  shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> cloud_sub_;
  shared_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> pose_sub_;
  shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub_;
  SynchronizerImagePose sync_image_pose_;
  SynchronizerCloudPose sync_cloud_pose_;
  SynchronizerImageOdom sync_image_odom_;

  ros::Publisher map_local_pub_, map_local_inflate_pub_, esdf_pub_, map_all_pub_, unknown_pub_,
      update_range_pub_, depth_pub_, map_free_all_pub_, cloud_vis_pub_, seen_pub_, expl_pub_, global_pub_, cover_pub_;
  ros::Timer esdf_timer_, vis_timer_;
  ros::Subscriber local_cloud_sub_, extrinsic_sub_, global_cloud_sub_;
  ros::Timer ikdtree_timer_, cloud_vis_timer_;

  // params, depth projection
  double cx_, cy_, fx_, fy_;
  double depth_filter_maxdist_, depth_filter_mindist_;
  int depth_filter_margin_;
  double k_depth_scaling_factor_;
  int skip_pixel_;
  string frame_id_;
  double ikdtree_downsample_resolution_;
  int pose_type_;
  
  // msg publication
  double esdf_slice_height_;
  double visualization_truncate_height_, visualization_truncate_low_;
  bool show_esdf_time_, show_occ_time_;
  bool show_all_map_;

  // data
  // flags of map state
  bool local_updated_, esdf_need_update_;
  // input
  Eigen::Vector3d camera_pos_;
  Eigen::Quaterniond camera_q_;
  unique_ptr<cv::Mat> depth_image_;
  vector<Eigen::Vector3d> proj_points_;
  int proj_points_cnt, env_proj_points_cnt;
  double fuse_time_, esdf_time_, max_fuse_time_, max_esdf_time_;
  int fuse_num_, esdf_num_;
  pcl::PointCloud<pcl::PointXYZ> point_cloud_, env_point_cloud_;

  normal_distribution<double> rand_noise_;
  default_random_engine eng_;

  ros::Time map_start_time_;

  sensor_msgs::PointCloud2 newest_cloud_;
  std::mutex cloud_mutex_;
  bool cloud_ready_ = false;
  double voxel_size_, max_dist_gap;
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter_;

  friend SDFMap;
  Eigen::Matrix4d cam2body_;
  Eigen::Matrix3d camera_r_m_;
};
}

#endif