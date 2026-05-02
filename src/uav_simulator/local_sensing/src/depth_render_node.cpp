#include <fstream>
#include <iostream>
#include <vector>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>

#include <pcl_conversions/pcl_conversions.h>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <Eigen/Eigen>

#include <backward.hpp>
namespace backward {
backward::SignalHandling sh;
}

//#include <cloud_banchmark/cloud_banchmarkConfig.h>
using namespace cv;
using namespace std;
using namespace Eigen;

cv::Mat depth_mat;

// camera param
int width, height;
double fx, fy, cx, cy;
double sensing_horizon, sensing_rate, estimation_rate;
double init_camera1_yaw, init_camera1_pitch;

ros::Publisher pub_depth, pub_camera1;
ros::Publisher pub_pose;
ros::Subscriber odom_sub;
ros::Subscriber global_map_sub;
ros::Subscriber camera_pose_sub, camera_extrinsic_sub, camera1_extrinsic_sub;
ros::Timer local_sensing_timer, estimation_timer, camera1_sensing_timer;

bool has_global_map(false);
bool has_odom(false);
bool has_camera_pose(false);
bool has_camera1_pose(false);

Matrix4d cam02body, pre_cam02body, cam02body1, pre_cam02body1;
Matrix4d cam2world, cam2world_odom, cam2world1;
Matrix4d body_pose;
Eigen::Quaterniond cam2world_quat, cam2world_quat1;
nav_msgs::Odometry odom_;
geometry_msgs::PoseStamped camera_pose_;

ros::Time last_odom_stamp = ros::TIME_MAX;
ros::Time last_cam_stamp = ros::TIME_MAX;
ros::Time last_cam1_stamp = ros::TIME_MAX;
Eigen::Vector3d last_pose_world;
pcl::PointCloud<pcl::PointXYZ> cloudIn;

// raycast param
double occlusion_dist = 0.1;
double min_dist = 0.5;
double render_pt_size = 1.0;

void odometryCallbck(const nav_msgs::Odometry& odom) {
  has_odom = true;
  odom_ = odom;
  Matrix4d Pose_receive = Matrix4d::Identity();

  Eigen::Vector3d request_position;
  Eigen::Quaterniond request_pose;
  request_position.x() = odom.pose.pose.position.x;
  request_position.y() = odom.pose.pose.position.y;
  request_position.z() = odom.pose.pose.position.z;
  request_pose.x() = odom.pose.pose.orientation.x;
  request_pose.y() = odom.pose.pose.orientation.y;
  request_pose.z() = odom.pose.pose.orientation.z;
  request_pose.w() = odom.pose.pose.orientation.w;
  Pose_receive.block<3, 3>(0, 0) = request_pose.toRotationMatrix();
  Pose_receive(0, 3) = request_position(0);
  Pose_receive(1, 3) = request_position(1);
  Pose_receive(2, 3) = request_position(2);

  body_pose = Pose_receive;

  last_pose_world(0) = odom.pose.pose.position.x;
  last_pose_world(1) = odom.pose.pose.position.y;
  last_pose_world(2) = odom.pose.pose.position.z;

  last_odom_stamp = odom.header.stamp;

  // convert to cam pose
  if (!has_camera_pose) {
    cam2world = body_pose * pre_cam02body;
    cam2world_quat = cam2world.block<3, 3>(0, 0);
  }
  if (!has_camera1_pose) {
    cam2world1 = body_pose * pre_cam02body1;
    cam2world_quat1 = cam2world1.block<3, 3>(0, 0);
  }
}

void pubCameraPose(const ros::TimerEvent& event) {
  pub_pose.publish(camera_pose_);
}

void subCameraPose(const geometry_msgs::PoseStamped camera_pose) {
  has_camera_pose = true;
  cam2world(0, 3) = camera_pose.pose.position.x;
  cam2world(1, 3) = camera_pose.pose.position.y;
  cam2world(2, 3) = camera_pose.pose.position.z;
  cam2world.block<3, 3>(0, 0) = Eigen::Quaterniond(camera_pose.pose.orientation.w, camera_pose.pose.orientation.x,
                                                        camera_pose.pose.orientation.y, camera_pose.pose.orientation.z)
                                           .toRotationMatrix();
  
  cam2world_quat = cam2world.block<3, 3>(0, 0);
  last_cam_stamp = camera_pose.header.stamp;
  camera_pose_ = camera_pose;
}

void pointCloudCallBack(const sensor_msgs::PointCloud2& pointcloud_map) {
  if (has_global_map) return;

  ROS_WARN("Global Pointcloud received..");
  // load global map
  // transform map to point cloud format
  pcl::fromROSMsg(pointcloud_map, cloudIn);
  printf("global map has points: %d.\n", cloudIn.points.size());
  has_global_map = true;
}

void renderDepth() {
  double this_time = ros::Time::now().toSec();

  // depth_mat = cv::Mat::zeros();
  depth_mat = cv::Mat::zeros(height, width, CV_32FC1);

  Eigen::Matrix4d Tcw = cam2world.inverse();
  Eigen::Matrix3d Rcw = Tcw.block<3, 3>(0, 0);
  Eigen::Vector3d tcw = Tcw.block<3, 1>(0, 3);

  auto t1 = ros::Time::now();

  Eigen::Vector3d pos = cam2world.block<3, 1>(0, 3);
  vector<Eigen::Vector3d> visible_pts;

  for (auto pt : cloudIn.points) {
    Eigen::Vector3d pw(pt.x, pt.y, pt.z);
    if ((pos - pw).norm() > sensing_horizon) continue;
    Eigen::Vector3d pc = Rcw * pw + tcw;
    if (pc[2] <= 0.0) continue;
    visible_pts.push_back(pc);
  }

  for (auto pc : visible_pts) {
    bool occluded = false;
    // raycast
    // for (auto inter_pc : visible_pts) {
    //   double len = pc.norm();
    //   Eigen::Vector3d dir = pc.normalized();
    //   double dist = inter_pc.cross(dir).norm();
    //   double proj = inter_pc.dot(dir);
    //   if (dist < occlusion_dist && proj > 0 && proj < len && (pc-inter_pc).norm() > min_dist) {
    //     occluded = true;
    //     break;
    //   }
    // }
    if (occluded) continue;

    float projected_x, projected_y;
    projected_x = pc[0] / pc[2] * fx + cx;
    projected_y = pc[1] / pc[2] * fy + cy;
    if (projected_x < 0 || projected_x >= width || projected_y < 0 || projected_y >= height)
      continue;

    // std::cout << "(u,v): " << projected_x << ", " << projected_y << endl;
    float dist = pc[2];
    int r = render_pt_size * fx / dist + 0.5;
    // std::cout << "r: " << r << std::endl;
    int min_x = max(int(projected_x - r), 0);
    int max_x = min(int(projected_x + r), width - 1);
    int min_y = max(int(projected_y - r), 0);
    int max_y = min(int(projected_y + r), height - 1);

    for (int to_x = min_x; to_x <= max_x; to_x++)
      for (int to_y = min_y; to_y <= max_y; to_y++) {
        // std::cout << "(u',v'): " << to_x << ", " << to_y << std::endl;
        float value = depth_mat.at<float>(to_y, to_x);
        if (value < 1e-3) {
          depth_mat.at<float>(to_y, to_x) = dist;
        } else {
          depth_mat.at<float>(to_y, to_x) = min(value, dist);
        }
      }
  }

  cv_bridge::CvImage out_msg;
  if (!has_camera_pose) {
    out_msg.header.stamp = last_odom_stamp;}
  else {
    out_msg.header.stamp = last_cam_stamp;}
  out_msg.header.frame_id = "world";
  out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  out_msg.image = depth_mat.clone();
  pub_depth.publish(out_msg.toImageMsg());
}

void renderSensedPoints(const ros::TimerEvent& event) {
  if (!has_global_map) return;
  renderDepth();
}

void renderCamera1SensedPoints(const ros::TimerEvent& event) {
  if (!has_global_map) return;
  double this_time = ros::Time::now().toSec();

  // depth_mat = cv::Mat::zeros();
  depth_mat = cv::Mat::zeros(height, width, CV_32FC1);

  Eigen::Matrix4d Tcw = cam2world1.inverse();
  Eigen::Matrix3d Rcw = Tcw.block<3, 3>(0, 0);
  Eigen::Vector3d tcw = Tcw.block<3, 1>(0, 3);

  auto t1 = ros::Time::now();

  Eigen::Vector3d pos = cam2world1.block<3, 1>(0, 3);
  vector<Eigen::Vector3d> visible_pts;

  for (auto pt : cloudIn.points) {
    Eigen::Vector3d pw(pt.x, pt.y, pt.z);
    // if ((pos - pw).norm() > sensing_horizon) continue;
    Eigen::Vector3d pc = Rcw * pw + tcw;
    if (pc[2] <= 0.0) continue;
    visible_pts.push_back(pc);
  }

  for (auto pc : visible_pts) {
    bool occluded = false;
    // raycast
    // for (auto inter_pc : visible_pts) {
    //   double len = pc.norm();
    //   Eigen::Vector3d dir = pc.normalized();
    //   double dist = inter_pc.cross(dir).norm();
    //   double proj = inter_pc.dot(dir);
    //   if (dist < occlusion_dist && proj > 0 && proj < len && (pc-inter_pc).norm() > min_dist) {
    //     occluded = true;
    //     break;
    //   }
    // }
    if (occluded) continue;

    float projected_x, projected_y;
    projected_x = pc[0] / pc[2] * fx + cx;
    projected_y = pc[1] / pc[2] * fy + cy;
    if (projected_x < 0 || projected_x >= width || projected_y < 0 || projected_y >= height)
      continue;

    // std::cout << "(u,v): " << projected_x << ", " << projected_y << endl;
    float dist = pc[2];
    int r = render_pt_size * fx / dist + 0.5;
    // std::cout << "r: " << r << std::endl;
    int min_x = max(int(projected_x - r), 0);
    int max_x = min(int(projected_x + r), width - 1);
    int min_y = max(int(projected_y - r), 0);
    int max_y = min(int(projected_y + r), height - 1);

    for (int to_x = min_x; to_x <= max_x; to_x++)
      for (int to_y = min_y; to_y <= max_y; to_y++) {
        // std::cout << "(u',v'): " << to_x << ", " << to_y << std::endl;
        float value = depth_mat.at<float>(to_y, to_x);
        if (value < 1e-3) {
          depth_mat.at<float>(to_y, to_x) = dist;
        } else {
          depth_mat.at<float>(to_y, to_x) = min(value, dist);
        }
      }
  }

  cv_bridge::CvImage out_msg;
  if (!has_camera1_pose) {
    out_msg.header.stamp = last_odom_stamp;}
  else {
    out_msg.header.stamp = last_cam1_stamp;}
  out_msg.header.frame_id = "world";
  out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
  out_msg.image = depth_mat.clone();
  pub_camera1.publish(out_msg.toImageMsg());
}

void subCameraExtrinsic(const nav_msgs::OdometryConstPtr &odom)
{ 
  has_camera_pose = true;
  Eigen::Quaterniond cam2body_q = Eigen::Quaterniond(odom->pose.pose.orientation.w,
                                                     odom->pose.pose.orientation.x,
                                                     odom->pose.pose.orientation.y,
                                                     odom->pose.pose.orientation.z);
  Eigen::Matrix3d cam2body_r_m = cam2body_q.toRotationMatrix();
  cam02body.block<3, 3>(0, 0) = cam2body_r_m;
  cam02body(0, 3) = odom->pose.pose.position.x;
  cam02body(1, 3) = odom->pose.pose.position.y;
  cam02body(2, 3) = odom->pose.pose.position.z;
  cam02body(3, 3) = 1.0;
  cam2world = body_pose * cam02body;
  cam2world_quat = cam2world.block<3, 3>(0, 0);
  last_cam_stamp = odom->header.stamp;
}

void subCamera1Extrinsic(const nav_msgs::OdometryConstPtr &odom)
{ 
  has_camera1_pose = true;
  Eigen::Quaterniond cam2body_q = Eigen::Quaterniond(odom->pose.pose.orientation.w,
                                                     odom->pose.pose.orientation.x,
                                                     odom->pose.pose.orientation.y,
                                                     odom->pose.pose.orientation.z);
  Eigen::Matrix3d cam2body_r_m = cam2body_q.toRotationMatrix();
  cam02body1.block<3, 3>(0, 0) = cam2body_r_m;
  cam02body1(0, 3) = odom->pose.pose.position.x;
  cam02body1(1, 3) = odom->pose.pose.position.y;
  cam02body1(2, 3) = odom->pose.pose.position.z;
  cam02body1(3, 3) = 1.0;
  cam2world1 = body_pose * cam02body1;
  cam2world_quat1 = cam2world1.block<3, 3>(0, 0);
  last_cam1_stamp = odom->header.stamp;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "pcl_render");
  ros::NodeHandle nh("~");

  nh.getParam("cam_width", width);
  nh.getParam("cam_height", height);
  nh.getParam("cam_fx", fx);
  nh.getParam("cam_fy", fy);
  nh.getParam("cam_cx", cx);
  nh.getParam("cam_cy", cy);
  nh.getParam("sensing_horizon", sensing_horizon);
  nh.getParam("sensing_rate", sensing_rate);
  nh.getParam("estimation_rate", estimation_rate);
  nh.getParam("render_point_size", render_pt_size);
  nh.param("init_camera1_yaw", init_camera1_yaw, 0.0);
  nh.param("init_camera1_pitch", init_camera1_pitch, 0.0);
  pre_cam02body << 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  Eigen::Matrix3d Rwb_y, Rwb_p;
  Rwb_y << cos(init_camera1_yaw), -sin(init_camera1_yaw), 0.0, sin(init_camera1_yaw), cos(init_camera1_yaw), 0.0, 0.0, 0.0, 1.0;
  Rwb_p << cos(init_camera1_pitch), 0.0, -sin(init_camera1_pitch), 0.0, 1.0, 0.0, sin(init_camera1_pitch), 0.0, cos(init_camera1_pitch);
  Eigen::Matrix4d T_wb = Eigen::Matrix4d::Identity();
  T_wb.block<3, 3>(0, 0) = Rwb_y * Rwb_p;
  pre_cam02body1 = T_wb * pre_cam02body;

  // init cam2world transformation
  cam2world = Matrix4d::Identity();
  // subscribe point cloud
  global_map_sub = nh.subscribe("global_map", 1, pointCloudCallBack);
  odom_sub = nh.subscribe("odometry", 50, odometryCallbck);
  // camera_pose_sub = nh.subscribe("/planning/camera_pose", 50, subCameraPose);
  camera_extrinsic_sub = nh.subscribe<nav_msgs::Odometry>("/vins_fusion/extrinsic", 10, subCameraExtrinsic);
  camera1_extrinsic_sub = nh.subscribe<nav_msgs::Odometry>("/camera1/extrinsic", 10, subCamera1Extrinsic);

  // publisher depth image and color image
  pub_depth = nh.advertise<sensor_msgs::Image>("/pcl_render_node/depth", 1000);
  pub_camera1 = nh.advertise<sensor_msgs::Image>("/pcl_render_node/camera1", 1000);
  // pub_pose = nh.advertise<geometry_msgs::PoseStamped>("/pcl_render_node/sensor_pose", 1000);

  double sensing_duration = 1.0 / sensing_rate;
  double estimate_duration = 1.0 / estimation_rate;

  local_sensing_timer = nh.createTimer(ros::Duration(sensing_duration), renderSensedPoints);
  camera1_sensing_timer = nh.createTimer(ros::Duration(sensing_duration), renderCamera1SensedPoints);
  estimation_timer = nh.createTimer(ros::Duration(estimate_duration), pubCameraPose);

  ros::Rate rate(100);
  bool status = ros::ok();
  while (status) {
    ros::spinOnce();
    status = ros::ok();
    rate.sleep();
  }
}