#include <ros/ros.h>
#include <ros/package.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Int32.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <sys/stat.h>
#include <string>
#include <iomanip>

using namespace std;

// Enum matching FastExplorationFSM state
enum EXPL_STATE { INIT, WAIT_TRIGGER, PLAN_TRAJ, PUB_TRAJ, EXEC_TRAJ, FINISH };

class TrajectoryRecorder {
private:
  ros::NodeHandle nh_;
  ros::Subscriber odom_sub_;
  ros::Subscriber cam0_ext_sub_;
  ros::Subscriber fsm_sub_;
  ros::Timer record_timer_;

  // Data buffers
  nav_msgs::Odometry last_odom_;
  nav_msgs::Odometry last_cam0_ext_;

  // Flags
  bool has_odom_;
  bool has_cam0_;

  // Recording State
  bool is_recording_;
  bool mission_started_;
  ros::Time recording_start_time_;

  // File output
  ofstream log_file_;
  string file_path_;

public:
  TrajectoryRecorder(ros::NodeHandle& nh) 
    : nh_(nh), is_recording_(false), mission_started_(false),
      has_odom_(false), has_cam0_(false) {

    // --- Parameters Configuration ---
    string save_dir;
    string file_name;
    string odom_topic, cam0_topic, fsm_topic;
    double record_rate;

    nh_.param<string>("save_dir", save_dir, "");
    nh_.param<string>("file_name", file_name, "traj_analysis_data.csv");
    nh_.param<string>("odom_topic", odom_topic, "/state_ukf/odom");
    nh_.param<string>("cam0_ext_topic", cam0_topic, "/vins_fusion/extrinsic");
    nh_.param<string>("fsm_topic", fsm_topic, "/planning/fsm_state");
    nh_.param<double>("record_rate", record_rate, 20.0);

    // Create directory
    if (save_dir.empty()) {
      save_dir = ros::package::getPath("exploration_manager") + "/data";
    }
    mkdir(save_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    file_path_ = save_dir + "/" + file_name;

    // Open file
    log_file_.open(file_path_, ios::out | ios::trunc);
    if (log_file_.is_open()) {
      // Write CSV Header
      log_file_ << "time_relative," 
                << "odom_x,odom_y,odom_z,odom_vx,odom_vy,odom_vz,odom_qw,odom_qx,odom_qy,odom_qz,"
                << "cam0_x,cam0_y,cam0_z,cam0_qw,cam0_qx,cam0_qy,cam0_qz,"
                << "\n";
      ROS_INFO_STREAM("[Recorder] File opened: " << file_path_ << ". Waiting for exploration trigger...");
    } else {
      ROS_ERROR_STREAM("[Recorder] Failed to open file: " << file_path_);
    }

    // --- Subscribers ---
    odom_sub_ = nh_.subscribe(odom_topic, 10, &TrajectoryRecorder::odomCallback, this);
    cam0_ext_sub_ = nh_.subscribe(cam0_topic, 10, &TrajectoryRecorder::cam0Callback, this);
    fsm_sub_ = nh_.subscribe(fsm_topic, 10, &TrajectoryRecorder::fsmStateCallback, this);

    // --- Timer ---
    record_timer_ = nh_.createTimer(ros::Duration(1.0 / record_rate), &TrajectoryRecorder::recordCallback, this);
  }

  ~TrajectoryRecorder() {
    if (log_file_.is_open()) log_file_.close();
  }

  void fsmStateCallback(const std_msgs::Int32ConstPtr& msg) {
    int state = msg->data;

    // Start recording when transitioning from WAIT_TRIGGER -> PLAN_TRAJ
    // We detect this if we haven't started yet and the state becomes PLAN_TRAJ (or any active state)
    if (!mission_started_ && state == PLAN_TRAJ) {
      mission_started_ = true;
      is_recording_ = true;
      recording_start_time_ = ros::Time::now();
      ROS_INFO("[Recorder] Exploration triggered. Recording started.");
    }

    // Stop recording when finished
    if (mission_started_ && state == FINISH) {
      is_recording_ = false;
      ROS_INFO("[Recorder] Exploration finished. Recording stopped.");
    }
  }

  void odomCallback(const nav_msgs::OdometryConstPtr& msg) {
    last_odom_ = *msg;
    has_odom_ = true;
  }

  void cam0Callback(const nav_msgs::OdometryConstPtr& msg) {
    last_cam0_ext_ = *msg;
    has_cam0_ = true;
  }

  void recordCallback(const ros::TimerEvent& e) {
    if (!is_recording_ || !log_file_.is_open() || !has_odom_ || !mission_started_) return;

    double relative_time = (ros::Time::now() - recording_start_time_).toSec();
    
    // Ensure relative time is non-negative
    if (relative_time < 0) relative_time = 0.0;

    // Odom Data
    const auto& pos = last_odom_.pose.pose.position;
    const auto& vel = last_odom_.twist.twist.linear;
    const auto& ori = last_odom_.pose.pose.orientation;

    double ox = pos.x, oy = pos.y, oz = pos.z;
    double ovx = vel.x, ovy = vel.y, ovz = vel.z;
    double oqw = ori.w, oqx = ori.x, oqy = ori.y, oqz = ori.z;

    // Cam0 Data
    double c0x = 0.0, c0y = 0.0, c0z = 0.0;
    double c0qw = 1.0, c0qx = 0.0, c0qy = 0.0, c0qz = 0.0;
    if (has_cam0_) {
      const auto& p = last_cam0_ext_.pose.pose.position;
      const auto& q = last_cam0_ext_.pose.pose.orientation;
      c0x = p.x; c0y = p.y; c0z = p.z;
      c0qw = q.w; c0qx = q.x; c0qy = q.y; c0qz = q.z;
    }

    // Write to file
    log_file_ << fixed << setprecision(6)
              << relative_time << ","
              << ox << "," << oy << "," << oz << "," << ovx << "," << ovy << "," << ovz << ","
              << oqw << "," << oqx << "," << oqy << "," << oqz << ","
              << c0x << "," << c0y << "," << c0z << "," << c0qw << "," << c0qx << "," << c0qy << "," << c0qz
              << "\n";

    log_file_.flush();
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "trajectory_recorder");
  ros::NodeHandle nh("~");
  TrajectoryRecorder recorder(nh);
  ros::spin();
  return 0;
}