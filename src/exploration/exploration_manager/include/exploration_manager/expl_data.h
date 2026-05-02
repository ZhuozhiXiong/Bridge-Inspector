#ifndef _EXPL_DATA_H_
#define _EXPL_DATA_H_

#include <Eigen/Eigen>
#include <vector>
#include <bspline/Bspline.h>

using std::vector;
using Eigen::Vector3d;

namespace fast_planner {
struct FSMData {
  // FSM data
  bool trigger_, have_odom_, static_state_;
  vector<string> state_str_;

  Eigen::Vector3d odom_pos_, odom_vel_;  // odometry state
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_, odom_pitch_;

  Eigen::Vector3d start_pt_, start_vel_, start_acc_, start_yaw_, start_pitch_;  // start state
  vector<Eigen::Vector3d> start_poss;
  bspline::Bspline newest_traj_;
};

struct FSMParam {
  double replan_thresh1_;
  double replan_thresh2_;
  double replan_thresh3_;
  double replan_time_;  // second
};

struct ExplorationData {
  vector<vector<Vector3d>> frontiers_, surfaces_, surface_frontiers_;
  vector<vector<Vector3d>> dead_frontiers_, dead_surfaces_;
  vector<pair<Vector3d, Vector3d>> frontier_boxes_;
  vector<Vector3d> points_, surface_points_;
  vector<Vector3d> averages_;
  vector<Vector3d> views_, surface_views_;
  vector<double> pitches_;
  vector<double> yaws_, surface_yaws_, surface_pitchs_;
  vector<Vector3d> global_tour_;
  vector<char> surface_states_;

  vector<int> refined_ids_;
  vector<vector<Vector3d>> n_points_;
  vector<Vector3d> unrefined_points_;
  vector<Vector3d> refined_points_;
  vector<Vector3d> refined_views_;  // points + dir(yaw)
  vector<Vector3d> refined_views1_, refined_views2_;
  vector<Vector3d> refined_tour_;

  Vector3d next_goal_;
  double next_yaw_, next_pitch_;
  vector<Vector3d> path_next_goal_;

  // viewpoint planning
  // vector<Vector4d> views_;
  vector<Vector3d> views_vis1_, views_vis2_;
  vector<Vector3d> centers_, scales_;

  // final viewpoints
  vector<Vector3d> updated_points_, updated_views_, updated_visib_cells_, all_points_;
  vector<int> updated_counts_, updated_iternums_;
  vector<int> updated_ids_;
  vector<Vector3d> updated_views1_, updated_views2_;

  // viewpoints cluster
  vector<Vector3d> cluster_viewpoints_;
  vector<Vector3d> cluster_averages_;
  vector<Vector3d> boundary_clusters_, all_boundary_clusters_;
  vector<vector<Vector3d>> drone_cluster_averages_;
  vector<int> cluster_ids_;
  int cluster_num_;

  vector<Vector3d> pts_normals_, pts_positions_, pts_views_;

  ros::Time surface_update_time_ = ros::Time::now();
};

struct ExplorationParam {
  // params
  bool refine_local_;
  int refined_num_;
  double refined_radius_;
  int top_view_num_;
  double max_decay_;
  string tsp_dir_;  // resource dir of tsp solver
  double relax_time_;
};

}  // namespace fast_planner

#endif