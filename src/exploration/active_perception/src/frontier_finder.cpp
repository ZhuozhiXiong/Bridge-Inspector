#include <active_perception/frontier_finder.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <plan_env/edt_environment.h>
#include <active_perception/perception_utils.h>
#include <active_perception/graph_node.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>

#include <Eigen/Eigenvalues>

namespace fast_planner {
FrontierFinder::FrontierFinder(const EDTEnvironment::Ptr& edt, ros::NodeHandle& nh) {
  this->edt_env_ = edt;
  int voxel_num = edt->sdf_map_->getVoxelNum();
  frontier_flag_ = vector<char>(voxel_num, 0);
  fill(frontier_flag_.begin(), frontier_flag_.end(), 0);

  nh.param("frontier/cluster_min", cluster_min_, -1);
  nh.param("frontier/cluster_size_xy", cluster_size_xy_, -1.0);
  nh.param("frontier/cluster_size_z", cluster_size_z_, -1.0);
  nh.param("frontier/cluster_size_xyz", cluster_size_xyz_, -1.0);
  nh.param("frontier/min_candidate_dist", min_candidate_dist_, -1.0);
  nh.param("frontier/min_candidate_clearance", min_candidate_clearance_, -1.0);
  nh.param("frontier/candidate_dphi", candidate_dphi_, -1.0);
  nh.param("frontier/candidate_dtheta", candidate_dtheta_, -1.0);
  nh.param("frontier/candidate_rmax", candidate_rmax_, -1.0);
  nh.param("frontier/candidate_rmin", candidate_rmin_, -1.0);
  nh.param("frontier/candidate_rnum", candidate_rnum_, -1);
  nh.param("frontier/candidate_hnum", candidate_hnum_, -1);
  nh.param("frontier/candidate_hmax", candidate_hmax_, -1.0);
  nh.param("frontier/candidate_hmin", candidate_hmin_, -1.0);
  nh.param("frontier/down_sample", down_sample_, -1);
  nh.param("frontier/min_visib_num", min_visib_num_, -1);
  nh.param("frontier/min_view_finish_fraction", min_view_finish_fraction_, -1.0);
  nh.param("frontier/max_tsp_num", tsp_num_, -1);
  nh.param("frontier/distance_penalty_coefficient", dis_plt_, -1.0);
  nh.param("frontier/heading_penalty_coefficient", heading_plt_, -1.0);
  nh.param("frontier/angle_penalty_coefficient", angle_plt_, -1.0);
  nh.param("frontier/normal_radius", normal_radius_, -1.0);
  nh.param("frontier/truncate_height", truncate_height_, -1.0);
  nh.param("frontier/truncate_low", truncate_low_, -1.0);
  nh.param("frontier/min_dist_to_obstacle", min_dist_to_obstacle_, -1.0);

  raycaster_.reset(new RayCaster);
  resolution_ = edt_env_->sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  edt_env_->sdf_map_->getRegion(origin, size);
  raycaster_->setParams(resolution_, origin);

  percep_utils_.reset(new PerceptionUtils(nh));
  cluster_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
}

FrontierFinder::~FrontierFinder() {
}

void FrontierFinder::searchFrontiers() {
  ros::Time t1 = ros::Time::now();
  tmp_frontiers_.clear();

  // Bounding box of updated region
  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max, true);

  // Removed changed frontiers in updated map
  auto resetFlag = [&](list<Frontier>::iterator& iter, list<Frontier>& frontiers) {
    Eigen::Vector3i idx;
    for (auto cell : iter->cells_) {
      edt_env_->sdf_map_->posToIndex(cell, idx);
      frontier_flag_[toadr(idx)] = 0;
    }
    iter = frontiers.erase(iter);
  };

  // std::cout << "Before remove: " << full_frontiers_.size() << std::endl;

  removed_ids_.clear();
  int rmv_idx = 0;
  for (auto iter = full_frontiers_.begin(); iter != full_frontiers_.end();) {
    if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) &&
        isFrontierChanged(*iter)) {
      resetFlag(iter, full_frontiers_);
      removed_ids_.push_back(rmv_idx);
    } else {
      ++rmv_idx;
      ++iter;
    }
  }
  // std::cout << "After remove: " << full_frontiers_.size() << std::endl;
  for (auto iter = dormant_frontiers_.begin(); iter != dormant_frontiers_.end();) {
    if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) &&
        isFrontierChanged(*iter))
      resetFlag(iter, dormant_frontiers_);
    else
      ++iter;
  }

  // Search new frontier within box slightly inflated from updated box
  Vector3d search_min = update_min - Vector3d(1, 1, 0.5);
  Vector3d search_max = update_max + Vector3d(1, 1, 0.5);
  Vector3d box_min, box_max;
  edt_env_->sdf_map_->getBox(box_min, box_max);
  for (int k = 0; k < 3; ++k) {
    search_min[k] = max(search_min[k], box_min[k]);
    search_max[k] = min(search_max[k], box_max[k]);
  }
  Eigen::Vector3i min_id, max_id;
  edt_env_->sdf_map_->posToIndex(search_min, min_id);
  edt_env_->sdf_map_->posToIndex(search_max, max_id);

  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z) {
        // Scanning the updated region to find seeds of frontiers
        Eigen::Vector3i cur(x, y, z);
        if (frontier_flag_[toadr(cur)] == 0 && satisfySurfaceFrontierCell(cur)) {
          // Expand from the seed cell to find a complete frontier cluster
          expandFrontier(cur);
        }
      }
  
  splitLargeFrontiers(tmp_frontiers_);
  // std::cout << "frontier_finder finds size of new frontier: ";
  // for (auto& fronter: tmp_frontiers_) {
  //   std::cout << fronter.cells_.size() << " ";
  // }
  // std::cout << std::endl;

  // ROS_WARN_THROTTLE(5.0, "Frontier t: %lf", (ros::Time::now() - t1).toSec());
  cout << "[Frontier] Frontier t: " << (ros::Time::now() - t1).toSec() << std::endl;
}

void FrontierFinder::expandFrontier(const Eigen::Vector3i& first) {
  // std::cout << "depth: " << depth << std::endl;
  auto t1 = ros::Time::now();

  // Data for clustering
  queue<Eigen::Vector3i> cell_queue;
  vector<Eigen::Vector3d> expanded;
  Vector3d pos;

  edt_env_->sdf_map_->indexToPos(first, pos);
  expanded.push_back(pos);
  cell_queue.push(first);
  frontier_flag_[toadr(first)] = 1;

  // Search frontier cluster based on region growing (distance clustering)
  while (!cell_queue.empty()) {
    auto cur = cell_queue.front();
    cell_queue.pop();
    auto nbrs = allNeighbors(cur);
    for (auto nbr : nbrs) {
      // Qualified cell should be inside bounding box and frontier cell not clustered
      int adr = toadr(nbr);
      if (frontier_flag_[adr] == 1 || !edt_env_->sdf_map_->isInBox(nbr) ||
          !satisfySurfaceFrontierCell(nbr))
        continue;

      edt_env_->sdf_map_->indexToPos(nbr, pos);
      if (pos[2] < truncate_low_ || pos[2] > truncate_height_) continue;  // Remove noise close to ground
      expanded.push_back(pos);
      cell_queue.push(nbr);
      frontier_flag_[adr] = 1;
    }
  }
  if (expanded.size() > cluster_min_) {
    // Compute detailed info
    Frontier frontier;
    frontier.cells_ = expanded;
    computeFrontierInfo(frontier);
    tmp_frontiers_.push_back(frontier);
  }
}

void FrontierFinder::splitLargeFrontiers(list<Frontier>& frontiers) {
  list<Frontier> splits, tmps;
  for (auto it = frontiers.begin(); it != frontiers.end(); ++it) {
    // Check if each frontier needs to be split horizontally
    if (splitFrontierIn3D(*it, splits)) {
      tmps.insert(tmps.end(), splits.begin(), splits.end());
      splits.clear();
    } else
      tmps.push_back(*it);
  }
  frontiers = tmps;
}

bool FrontierFinder::splitFrontierIn3D(const Frontier& frontier, list<Frontier>& splits)
{
  auto mean = frontier.average_;
  bool need_split = false;
  for (auto cell : frontier.filtered_cells_) {
    if ((cell - mean).norm() > cluster_size_xyz_) {
      need_split = true;
      break;
    }
  }
  if (!need_split)
    return false;

  // Compute covariance matrix of cells
  Eigen::Matrix3d cov;
  cov.setZero();
  for (auto cell : frontier.filtered_cells_) {
    Eigen::Vector3d diff = cell - mean;
    cov += diff * diff.transpose();
  }
  cov /= double(frontier.filtered_cells_.size());

  // Find eigenvector corresponds to maximal eigenvalue
  Eigen::EigenSolver<Eigen::Matrix3d> es(cov);
  auto values = es.eigenvalues().real();
  auto vectors = es.eigenvectors().real();
  int max_idx;
  double max_eigenvalue = -1000000;
  for (int i = 0; i < values.rows(); ++i) {
    if (values[i] > max_eigenvalue) {
      max_idx = i;
      max_eigenvalue = values[i];
    }
  }
  Eigen::Vector3d first_pc = vectors.col(max_idx);

  // Split the frontier into two groups along the first PC
  Frontier ftr1, ftr2;
  for (auto cell : frontier.cells_) {
    if ((cell - mean).dot(first_pc) >= 0)
      ftr1.cells_.push_back(cell);
    else
      ftr2.cells_.push_back(cell);
  }
  computeFrontierInfo(ftr1);
  computeFrontierInfo(ftr2);

  // Recursive call to split frontier that is still too large
  list<Frontier> splits2;
  if (splitFrontierIn3D(ftr1, splits2)) {
    splits.insert(splits.end(), splits2.begin(), splits2.end());
    splits2.clear();
  }
  else
    splits.push_back(ftr1);

  if (splitFrontierIn3D(ftr2, splits2))
    splits.insert(splits.end(), splits2.begin(), splits2.end());
  else
    splits.push_back(ftr2);

  return true;
}

bool FrontierFinder::splitHorizontally(const Frontier& frontier, list<Frontier>& splits) {
  // Split a frontier into small piece if it is too large
  auto mean = frontier.average_.head<2>();
  bool need_split = false;
  for (auto cell : frontier.filtered_cells_) {
    if ((cell.head<2>() - mean).norm() > cluster_size_xy_) {
      need_split = true;
      break;
    }
  }
  if (!need_split) return false;

  // Compute principal component
  // Covariance matrix of cells
  Eigen::Matrix2d cov;
  cov.setZero();
  for (auto cell : frontier.filtered_cells_) {
    Eigen::Vector2d diff = cell.head<2>() - mean;
    cov += diff * diff.transpose();
  }
  cov /= double(frontier.filtered_cells_.size());

  // Find eigenvector corresponds to maximal eigenvector
  Eigen::EigenSolver<Eigen::Matrix2d> es(cov);
  auto values = es.eigenvalues().real();
  auto vectors = es.eigenvectors().real();
  int max_idx;
  double max_eigenvalue = -1000000;
  for (int i = 0; i < values.rows(); ++i) {
    if (values[i] > max_eigenvalue) {
      max_idx = i;
      max_eigenvalue = values[i];
    }
  }
  Eigen::Vector2d first_pc = vectors.col(max_idx);
  std::cout << "max idx: " << max_idx << std::endl;
  std::cout << "mean: " << mean.transpose() << ", first pc: " << first_pc.transpose() << std::endl;

  // Split the frontier into two groups along the first PC
  Frontier ftr1, ftr2;
  for (auto cell : frontier.cells_) {
    if ((cell.head<2>() - mean).dot(first_pc) >= 0)
      ftr1.cells_.push_back(cell);
    else
      ftr2.cells_.push_back(cell);
  }
  computeFrontierInfo(ftr1);
  computeFrontierInfo(ftr2);

  // Recursive call to split frontier that is still too large
  list<Frontier> splits2;
  if (splitHorizontally(ftr1, splits2)) {
    splits.insert(splits.end(), splits2.begin(), splits2.end());
    splits2.clear();
  } else
    splits.push_back(ftr1);

  if (splitHorizontally(ftr2, splits2))
    splits.insert(splits.end(), splits2.begin(), splits2.end());
  else
    splits.push_back(ftr2);

  return true;
}

bool FrontierFinder::isInBoxes(
    const vector<pair<Vector3d, Vector3d>>& boxes, const Eigen::Vector3i& idx) {
  Vector3d pt;
  edt_env_->sdf_map_->indexToPos(idx, pt);
  for (auto box : boxes) {
    // Check if contained by a box
    bool inbox = true;
    for (int i = 0; i < 3; ++i) {
      inbox = inbox && pt[i] > box.first[i] && pt[i] < box.second[i];
      if (!inbox) break;
    }
    if (inbox) return true;
  }
  return false;
}

void FrontierFinder::updateFrontierCostMatrix() {
  auto updateCost = [](const list<Frontier>::iterator& it1, const list<Frontier>::iterator& it2) {
    // Search path from old cluster's top viewpoint to new cluster'
    Viewpoint& vui = it1->viewpoints_.front();
    Viewpoint& vuj = it2->viewpoints_.front();
    vector<Vector3d> path_ij;
    double cost_ij = ViewNode::computeCost(
        vui.pos_, vuj.pos_, vui.pitch_, vuj.pitch_, vui.yaw_, vuj.yaw_, Vector3d(0, 0, 0), 0, 0, path_ij);
    // Insert item for both old and new clusters
    it1->costs_.push_back(cost_ij);
    it1->paths_.push_back(path_ij);
    reverse(path_ij.begin(), path_ij.end());
    it2->costs_.push_back(cost_ij);
    it2->paths_.push_back(path_ij);
  };

  // Compute path and cost in frontiers_
  for (auto it1 = frontiers_.begin(); it1 != frontiers_.end(); ++it1)
    for (auto it2 = it1; it2 != frontiers_.end(); ++it2) {
      if (it1 == it2) {
        it1->costs_.push_back(0);
        it1->paths_.push_back({});
      } else
        updateCost(it1, it2);
    }
}

void FrontierFinder::mergeFrontiers(Frontier& ftr1, const Frontier& ftr2) {
  // Merge ftr2 into ftr1
  ftr1.average_ =
      (ftr1.average_ * double(ftr1.cells_.size()) + ftr2.average_ * double(ftr2.cells_.size())) /
      (double(ftr1.cells_.size() + ftr2.cells_.size()));
  ftr1.cells_.insert(ftr1.cells_.end(), ftr2.cells_.begin(), ftr2.cells_.end());
  computeFrontierInfo(ftr1);
}

bool FrontierFinder::canBeMerged(const Frontier& ftr1, const Frontier& ftr2) {
  Vector3d merged_avg =
      (ftr1.average_ * double(ftr1.cells_.size()) + ftr2.average_ * double(ftr2.cells_.size())) /
      (double(ftr1.cells_.size() + ftr2.cells_.size()));
  // Check if it can merge two frontier without exceeding size limit
  for (auto c1 : ftr1.cells_) {
    auto diff = c1 - merged_avg;
    if (diff.head<2>().norm() > cluster_size_xy_ || diff[2] > cluster_size_z_) return false;
  }
  for (auto c2 : ftr2.cells_) {
    auto diff = c2 - merged_avg;
    if (diff.head<2>().norm() > cluster_size_xy_ || diff[2] > cluster_size_z_) return false;
  }
  return true;
}

bool FrontierFinder::haveOverlap(
    const Vector3d& min1, const Vector3d& max1, const Vector3d& min2, const Vector3d& max2) {
  // Check if two box have overlap part
  Vector3d bmin, bmax;
  for (int i = 0; i < 3; ++i) {
    bmin[i] = max(min1[i], min2[i]);
    bmax[i] = min(max1[i], max2[i]);
    if (bmin[i] > bmax[i] + 1e-3) return false;
  }
  return true;
}

bool FrontierFinder::isFrontierChanged(const Frontier& ft) {
  for (auto cell : ft.cells_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!satisfySurfaceFrontierCell(idx)) return true;
  }
  return false;
}

void FrontierFinder::computeFrontierInfo(Frontier& ftr) {
  // Compute average position and bounding box of cluster
  ftr.average_.setZero();
  ftr.box_max_ = ftr.cells_.front();
  ftr.box_min_ = ftr.cells_.front();
  for (auto cell : ftr.cells_) {
    ftr.average_ += cell;
    for (int i = 0; i < 3; ++i) {
      ftr.box_min_[i] = min(ftr.box_min_[i], cell[i]);
      ftr.box_max_[i] = max(ftr.box_max_[i], cell[i]);
    }
  }
  ftr.average_ /= double(ftr.cells_.size());

  // Compute downsampled cluster
  downsample(ftr.cells_, ftr.filtered_cells_);
}

void FrontierFinder::computeFrontiersToVisit(Vector3d pos, Vector3d vel) {
  double t1 = ros::Time::now().toSec();
  pts_normals_.clear(); pts_positions_.clear();
  pos_ = pos;
  int new_num = 0;
  int new_dormant_num = 0;
  // Try find viewpoints for each cluster and categorize them according to viewpoint number
  for (auto& tmp_ftr : tmp_frontiers_) {
    // Search viewpoints around frontier
    sampleViewpointsIn3D(tmp_ftr, pos);
    if (!tmp_ftr.viewpoints_.empty()) {
      ++new_num;
      list<Frontier>::iterator inserted = full_frontiers_.insert(full_frontiers_.end(), tmp_ftr);
      // Sort the viewpoints by coverage fraction, best view in front
      sort(
          inserted->viewpoints_.begin(), inserted->viewpoints_.end(),
          [](const Viewpoint& v1, const Viewpoint& v2) { return v1.view_score_ > v2.view_score_; });
    } else {
      // Find no viewpoint, move cluster to dormant list
      dormant_frontiers_.push_back(tmp_ftr);
      ++new_dormant_num;
    }
  }
  
  vector<Frontier> filter_frontiers_;
  for (auto& ftr : full_frontiers_) {
    Viewpoint& vp = ftr.viewpoints_.front();

    Vector3d dir_to_view = (vp.pos_ - pos).normalized();
    Vector3d vel_dir = vel.normalized();
    double cos_theta = vel_dir.dot(dir_to_view);
    cos_theta = min(cos_theta, 0.0);  // Only penalize when heading is opposite to view direction
    double heading_penalty = exp(heading_plt_ * cos_theta);

    vp.dis_ = (vp.pos_ - pos_).norm();
    vp.score_ = vp.visib_num_ * exp(- dis_plt_ * vp.dis_) * heading_penalty;
    filter_frontiers_.push_back(ftr);
  }
  sort(filter_frontiers_.begin(), filter_frontiers_.end(),
    [](const Frontier& f1, const Frontier& f2) {
    return f1.viewpoints_.front().score_ > f2.viewpoints_.front().score_;});
  
  int filter_num = min(int(filter_frontiers_.size()), tsp_num_);
  frontiers_.clear();
  for (int i = 0; i < filter_num; ++i) {
    Frontier tmp_ftr = filter_frontiers_[i];
    list<Frontier>::iterator inserted = frontiers_.insert(frontiers_.end(), tmp_ftr);
  }
  
  int idx = 0;
  for (auto& ft : frontiers_) {
    ft.id_ = idx++;
  }

  // std::cout << "\nnew num: " << new_num << ", new dormant: " << new_dormant_num << std::endl;
  std::cout << "to visit: " << frontiers_.size() << ", dormant: " << dormant_frontiers_.size() << std::endl;
  cout << "[Frontier] Compute frontiers to visit t: " << ros::Time::now().toSec() - t1 << std::endl;
}

void FrontierFinder::getTopViewpointsInfo(
    const Vector3d& cur_pos, vector<Eigen::Vector3d>& points, vector<double>& pitches, vector<double>& yaws,
    vector<Eigen::Vector3d>& averages) {
  points.clear();
  pitches.clear();
  yaws.clear();
  averages.clear();
  for (auto frontier : full_frontiers_) {
    bool no_view = true;
    for (auto view : frontier.viewpoints_) {
      // Retrieve the first viewpoint that is far enough and has highest coverage
      if ((view.pos_ - cur_pos).norm() < min_candidate_dist_) continue;
      points.push_back(view.pos_);
      pitches.push_back(view.pitch_);
      yaws.push_back(view.yaw_);
      averages.push_back(frontier.average_);
      no_view = false;
      break;
    }
    if (no_view) {
      // All viewpoints are very close, just use the first one (with highest coverage).
      auto view = frontier.viewpoints_.front();
      points.push_back(view.pos_);
      pitches.push_back(view.pitch_);
      yaws.push_back(view.yaw_);
      averages.push_back(frontier.average_);
    }
  }
}

void FrontierFinder::getViewpointsInfo(
    const Vector3d& cur_pos, const vector<int>& ids, const int& view_num, const double& max_decay,
    vector<vector<Eigen::Vector3d>>& points, vector<vector<double>>& pitches, vector<vector<double>>& yaws) {
  points.clear();
  yaws.clear();
  pitches.clear();
  for (auto id : ids) {
    // Scan all frontiers to find one with the same id
    for (auto frontier : frontiers_) {
      if (frontier.id_ == id) {
        // Get several top viewpoints that are far enough
        vector<Eigen::Vector3d> pts;
        vector<double> ys;
        vector<double> ps;
        int visib_thresh = frontier.viewpoints_.front().visib_num_ * max_decay;
        for (auto view : frontier.viewpoints_) {
          if (pts.size() >= view_num || view.visib_num_ <= visib_thresh) break;
          if ((view.pos_ - cur_pos).norm() < min_candidate_dist_) continue;
          pts.push_back(view.pos_);
          ys.push_back(view.yaw_);
          ps.push_back(view.pitch_);
        }
        if (pts.empty()) {
          // All viewpoints are very close, ignore the distance limit
          for (auto view : frontier.viewpoints_) {
            if (pts.size() >= view_num || view.visib_num_ <= visib_thresh) break;
            pts.push_back(view.pos_);
            ys.push_back(view.yaw_);
            ps.push_back(view.pitch_);
          }
        }
        points.push_back(pts);
        yaws.push_back(ys);
        pitches.push_back(ps);
      }
    }
  }
}

void FrontierFinder::getFrontiers(vector<vector<Eigen::Vector3d>>& clusters) {
  clusters.clear();
  for (auto frontier : full_frontiers_)
    clusters.push_back(frontier.cells_);
  // clusters.push_back(frontier.filtered_cells_);
}

void FrontierFinder::getDormantFrontiers(vector<vector<Vector3d>>& clusters) {
  clusters.clear();
  for (auto ft : dormant_frontiers_)
    clusters.push_back(ft.cells_);
}

void FrontierFinder::getFrontierBoxes(vector<pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes) {
  boxes.clear();
  for (auto frontier : full_frontiers_) {
    Vector3d center = (frontier.box_max_ + frontier.box_min_) * 0.5;
    Vector3d scale = frontier.box_max_ - frontier.box_min_;
    boxes.push_back(make_pair(center, scale));
  }
}

void FrontierFinder::getPathForTour(
    const Vector3d& pos, const vector<int>& frontier_ids, vector<Vector3d>& path) {
  // Make an frontier_indexer to access the frontier list easier
  vector<list<Frontier>::iterator> frontier_indexer;
  for (auto it = frontiers_.begin(); it != frontiers_.end(); ++it)
    frontier_indexer.push_back(it);

  // Compute the path from current pos to the first frontier
  vector<Vector3d> segment;
  ViewNode::searchPath(pos, frontier_indexer[frontier_ids[0]]->viewpoints_.front().pos_, segment);
  path.insert(path.end(), segment.begin(), segment.end());

  // Get paths of tour passing all clusters
  for (int i = 0; i < frontier_ids.size() - 1; ++i) {
    // Move to path to next cluster
    auto path_iter = frontier_indexer[frontier_ids[i]]->paths_.begin();
    int next_idx = frontier_ids[i + 1];
    for (int j = 0; j < next_idx; ++j)
      ++path_iter;
    path.insert(path.end(), path_iter->begin(), path_iter->end());
  }
}

void FrontierFinder::getFullCostMatrix(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_pitch, const Vector3d cur_yaw,
    Eigen::MatrixXd& mat) {
  if (false) {
    // Use symmetric TSP formulation
    int dim = frontiers_.size() + 2;
    mat.resize(dim, dim);  // current pose (0), sites, and virtual depot finally

    int i = 1, j = 1;
    for (auto ftr : frontiers_) {
      for (auto cs : ftr.costs_)
        mat(i, j++) = cs;
      ++i;
      j = 1;
    }

    // Costs from current pose to sites
    for (auto ftr : frontiers_) {
      Viewpoint vj = ftr.viewpoints_.front();
      vector<Vector3d> path;
      mat(0, j) = mat(j, 0) =
          ViewNode::computeCost(cur_pos, vj.pos_, cur_yaw[0], vj.yaw_, 
            cur_pitch[0], vj.pitch_, cur_vel, cur_pitch[1], cur_yaw[1], path);
      ++j;
    }
    // Costs from depot to sites, the same large vaule
    for (j = 1; j < dim - 1; ++j) {
      mat(dim - 1, j) = mat(j, dim - 1) = 100;
    }
    // Zero cost to depot to ensure connection
    mat(0, dim - 1) = mat(dim - 1, 0) = -10000;

  } else {
    // Use Asymmetric TSP
    int dimen = frontiers_.size();
    mat.resize(dimen + 1, dimen + 1);
    std::cout << "ATSP mat size: " << mat.rows() << ", " << mat.cols() << std::endl;
    // Fill block for clusters
    int i = 1, j = 1;
    for (auto ftr : frontiers_) {
      for (auto cs : ftr.costs_) {
        mat(i, j++) = cs;
      }
      ++i;
      j = 1;
    }

    // Fill block from current state to clusters
    mat.leftCols<1>().setZero();
    for (auto ftr : frontiers_) {
      Viewpoint vj = ftr.viewpoints_.front();
      vector<Vector3d> path;
      mat(0, j++) =
          ViewNode::computeCost(cur_pos, vj.pos_, cur_yaw[0], vj.yaw_, 
            cur_pitch[0], vj.pitch_, cur_vel, cur_pitch[1], cur_yaw[1], path);
    }
  }
}

void FrontierFinder::findViewpoints(
    const Vector3d& sample, const Vector3d& ftr_avg, vector<Viewpoint>& vps) {
  if (!edt_env_->sdf_map_->isInBox(sample) ||
      edt_env_->sdf_map_->getInflateOccupancy(sample) == 1 || isNearUnknown(sample))
    return;

  double left_angle_, right_angle_, vertical_angle_, ray_length_;

  // Central yaw is determined by frontier's average position and sample
  auto dir = ftr_avg - sample;
  double hc = atan2(dir[1], dir[0]);

  vector<int> slice_gains;
  // Evaluate info gain of different slices
  for (double phi_h = -M_PI_2; phi_h <= M_PI_2 + 1e-3; phi_h += M_PI / 18) {
    // Compute gain of one slice
    int gain = 0;
    for (double phi_v = -vertical_angle_; phi_v <= vertical_angle_; phi_v += vertical_angle_ / 3) {
      // Find endpoint of a ray
      Vector3d end;
      end[0] = sample[0] + ray_length_ * cos(phi_v) * cos(hc + phi_h);
      end[1] = sample[1] + ray_length_ * cos(phi_v) * sin(hc + phi_h);
      end[2] = sample[2] + ray_length_ * sin(phi_v);

      // Do raycasting to check info gain
      Vector3i idx;
      raycaster_->input(sample, end);
      while (raycaster_->nextId(idx)) {
        // Hit obstacle, stop the ray
        if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 || !edt_env_->sdf_map_->isInBox(idx))
          break;
        // Count number of unknown cells
        if (edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) ++gain;
      }
    }
    slice_gains.push_back(gain);
  }

  // Sum up slices' gain to get different yaw's gain
  vector<pair<double, int>> yaw_gains;
  for (int i = 0; i < 6; ++i)  // [-90,-10]-> [10,90], delta_yaw = 20, 6 groups
  {
    double yaw = hc - M_PI_2 + M_PI / 9.0 * i + right_angle_;
    int gain = 0;
    for (int j = 2 * i; j < 2 * i + 9; ++j)  // 80 degree hFOV, 9 slices
      gain += slice_gains[j];
    yaw_gains.push_back(make_pair(yaw, gain));
  }

  // Get several yaws with highest gain
  vps.clear();
  sort(
      yaw_gains.begin(), yaw_gains.end(),
      [](const pair<double, int>& p1, const pair<double, int>& p2) {
        return p1.second > p2.second;
      });
  for (int i = 0; i < 3; ++i) {
    if (yaw_gains[i].second < min_visib_num_) break;
    Viewpoint vp = { sample, yaw_gains[i].first, yaw_gains[i].second };
    while (vp.yaw_ < -M_PI)
      vp.yaw_ += 2 * M_PI;
    while (vp.yaw_ > M_PI)
      vp.yaw_ -= 2 * M_PI;
    vps.push_back(vp);
  }
}

// Sample viewpoints around frontier's average position, check coverage to the frontier cells
void FrontierFinder::sampleViewpointsIn3D(Frontier& frontier, Vector3d& pos)
{ 
  // Compute average normal of a frontier
  Vector3d avg_normal, normal_view;
  // double normal_pitch, normal_yaw;
  bool success = false;
  // success = computeNormal(frontier, pos, avg_normal);
  // normal_view = -avg_normal;
  // normal_pitch = asin(normal_view[2]);
  // normal_yaw = atan2(normal_view[1], normal_view[0]);

  // Evaluate sample viewpoints on circles, find ones that cover most cells
  for (double rc = candidate_rmax_, dr = (candidate_rmax_ - candidate_rmin_) / (double)candidate_rnum_;
       rc >= candidate_rmin_ - 1e-3; rc -= dr)
    for (double phi = -M_PI; phi < M_PI; phi += candidate_dphi_) {
      for (double theta = -M_PI/3; theta <= M_PI/3 + 1e-3; theta += candidate_dtheta_) {
        const Vector3d dir = Vector3d(cos(theta)*cos(phi), cos(theta)*sin(phi), sin(theta));
        const Vector3d sample_pos = frontier.average_ + rc * dir;

        // Qualified viewpoint is in bounding box and in safe region
        if (!edt_env_->sdf_map_->isInBox(sample_pos) ||
            edt_env_->sdf_map_->getInflateOccupancy(sample_pos) == 1)
          continue;
        
        double dist_to_obstacle = edt_env_->sdf_map_->getDistToObstacle(sample_pos);
        if (dist_to_obstacle < min_dist_to_obstacle_) continue;
        double dist_to_unknown = edt_env_->sdf_map_->getDistance(sample_pos);
        if (dist_to_unknown < min_candidate_clearance_) continue;
        
        double avg_yaw, avg_pitch, dist, score;

        // Calculate average direction to frontier cells
        // auto& cells = frontier.filtered_cells_;
        // for (size_t i = 1; i < cells.size(); ++i) {
        //   Eigen::Vector3d cell = cells[i];
        //   Eigen::Vector3d dir = (cell - sample_pos).normalized();
        //   Vector2d dir_py = getPitchYaw(dir);
        //   avg_yaw = avg_yaw + dir_py(1);
        //   avg_pitch = avg_pitch + dir_py(0);
        // }
        // // avg_pitch = avg_pitch / cells.size();
        // avg_pitch = 0.0;
        // avg_yaw = avg_yaw / cells.size();
        // Vector3d sample_dir = Vector3d(cos(avg_pitch)*cos(avg_yaw), cos(avg_pitch)*sin(avg_yaw), sin(avg_pitch));

        // Viewpoint direction is pointed to the center of frontier
        Vector3d center = frontier.average_;
        Vector3d center_dir = (center - sample_pos).normalized();
        Vector2d dir_py = getPitchYaw(center_dir);
        avg_pitch = dir_py(0);
        avg_yaw = dir_py(1);
        Vector3d sample_dir = Vector3d(cos(avg_pitch)*cos(avg_yaw), cos(avg_pitch)*sin(avg_yaw), sin(avg_pitch));

        auto& raw_cells = frontier.cells_;
        vector<Vector3d> visib_cells;
        // Compute the fraction of covered and visible cells
        int visib_num = countVisibleFrontierCells(sample_pos, avg_pitch, avg_yaw, raw_cells, visib_cells);
        if (visib_num > min_visib_num_) {
          double view_score;
          if (success) {
            double angle_diff = acos(sample_dir.dot(-avg_normal));
            angle_diff = min(angle_diff, M_PI - angle_diff);
            view_score = visib_num * exp(- angle_plt_ * angle_diff);
          } else { view_score = visib_num; }
          
          // double dist = (sample_pos - pos_).norm();
          // double score = visib_num * exp(- dis_plt_ * dist);
          Viewpoint vp = { sample_pos, avg_pitch, avg_yaw, visib_num, view_score, dist, score };
          frontier.viewpoints_.push_back(vp);
        }
      }
    }
}

bool FrontierFinder::computeNormal(const Frontier& frontier, Vector3d& avg_normal)
{
  // Search point clouds in the box
  cluster_pts.reset(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto cell : frontier.cells_) {
    Vector3d bmin, bmax;
    getCellBox(cell, bmin, bmax);
    BoxPointType cell_box;
    for (int i = 0; i < 3; i++) {
      cell_box.vertex_min[i] = static_cast<float>(bmin[i]);
      cell_box.vertex_max[i] = static_cast<float>(bmax[i]);
    }
    PointVector pts;
    edt_env_->sdf_map_->kdtreeBoxSearch(cell_box, pts);
    cluster_pts->insert(cluster_pts->end(), pts.begin(), pts.end());
  }
  cout << "cluster pts size: " << cluster_pts->size() << endl;

  if (cluster_pts->empty()) {
  avg_normal = Eigen::Vector3d(0, 0, 1);
  return false;
  }

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cluster_pts);

  auto& cells = frontier.filtered_cells_;
  avg_normal = Vector3d::Zero();
  vector<int> pt_ids;
  for (auto& cell: cells) {
    pcl::PointXYZ point;
    point.x = cell.x();
    point.y = cell.y();
    point.z = cell.z();
    std::vector<int> nearest_indices;
    std::vector<float> nearest_distances;
    kdtree.nearestKSearch(point, 1, nearest_indices, nearest_distances);
    pt_ids.push_back(nearest_indices[0]);
  }
  cout << "normal size :" << pt_ids.size() << endl;

  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud(cluster_pts);

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(normal_radius_);

  ne.setIndices(boost::make_shared<std::vector<int>>(pt_ids));
  pcl::PointCloud<pcl::Normal>::Ptr pt_normals(new pcl::PointCloud<pcl::Normal>());
  ne.compute(*pt_normals);
  
  // map<int, Eigen::Vector3d> pt_idx_normal_pairs_;
  vector<Vector3d> pts_normals, pts_positions;
  for (int i = 0; i < (int)pt_ids.size(); ++i) {
    // int unc_id = pt_ids[i];
    Eigen::Vector3d normal_vector;
    normal_vector(0) = pt_normals->points[i].normal_x;
    normal_vector(1) = pt_normals->points[i].normal_y;
    normal_vector(2) = pt_normals->points[i].normal_z;

    // pt_idx_normal_pairs_[unc_id] = normal_vector.normalized();
    if (normal_vector.hasNaN()) continue;
    pts_normals.push_back(normal_vector.normalized());
    pts_positions.push_back(cluster_pts->points[pt_ids[i]].getVector3fMap().cast<double>());
    pts_normals_.push_back(normal_vector.normalized());
    pts_positions_.push_back(cluster_pts->points[pt_ids[i]].getVector3fMap().cast<double>());
    avg_normal += normal_vector.normalized();
  }

  if (avg_normal.norm() < 1e-6) {
    avg_normal = Eigen::Vector3d(0, 0, 1);
    return false;
  } else {
    avg_normal.normalize();
  }
  return true;
}

void FrontierFinder::getFrontierNormal(vector<Vector3d>& normals, vector<Vector3d>& positions, vector<Vector3d>& views)
{
  if (pts_normals_.empty()) return;
  vector<Vector3d> pts_views_;
  for (int i = 0; i < pts_normals_.size(); ++i) {
    pts_views_.push_back(pts_positions_[i] + 0.5 * pts_normals_[i]);
  }
  normals = pts_normals_;
  positions = pts_positions_;
  views = pts_views_;
}

inline void FrontierFinder::getCellBox(const Vector3d& pos, Vector3d& bmin, Vector3d& bmax)
{
  bmin = pos - Vector3d(resolution_, resolution_, resolution_) / 2.0;
  bmax = pos + Vector3d(resolution_, resolution_, resolution_) / 2.0;
}

inline Eigen::Vector2d FrontierFinder::getPitchYaw(const Eigen::Vector3d& vec)
{
  Eigen::Vector2d PY;
  double pitch = std::asin(vec.z() / (vec.norm() + 1e-3));  // calculate pitch angle
  double yaw = std::atan2(vec.y(), vec.x());                // calculate yaw angle

  PY(0) = pitch;
  PY(1) = yaw;

  return PY;
}

inline void FrontierFinder::wrapAngle(double& angle)
{
  while (angle < -M_PI) angle += 2 * M_PI;
  while (angle > M_PI) angle -= 2 * M_PI;
}

int FrontierFinder::countVisibleFrontierCells(const Eigen::Vector3d& pos, const double& pitch, const double& yaw,
    const vector<Eigen::Vector3d>& cluster, vector<Eigen::Vector3d>& visib_cells)
{
  visib_cells.clear();
  percep_utils_->setPose_PY(pos, pitch, yaw);
  int visib_num = 0;
  Eigen::Vector3i idx;
  for (auto cell : cluster) {
    // Check if frontier cell is inside FOV
    if (!percep_utils_->insideFOV(cell))
      continue;

    // Check if frontier cell is visible (not occulded by obstacles)
    raycaster_->input(cell, pos);
    bool visib = true;
    while (raycaster_->nextId(idx)) {
      if (edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::OCCUPIED ||
          edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
        visib = false;
        break;
      }
    }
    if (visib) {
      visib_num += 1;
      visib_cells.push_back(cell);
    }
  }
  return visib_num;
}

bool FrontierFinder::isFrontierCovered() {
  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max);

  auto checkChanges = [&](const list<Frontier>& frontiers) {
    for (auto ftr : frontiers) {
      if (!haveOverlap(ftr.box_min_, ftr.box_max_, update_min, update_max)) continue;
      const int change_thresh = min_view_finish_fraction_ * ftr.cells_.size();
      int change_num = 0;
      for (auto cell : ftr.cells_) {
        Eigen::Vector3i idx;
        edt_env_->sdf_map_->posToIndex(cell, idx);
        if (!satisfySurfaceFrontierCell(idx) && ++change_num >= change_thresh)
          return true;
      }
    }
    return false;
  };

  if (checkChanges(frontiers_) || checkChanges(dormant_frontiers_)) return true;

  return false;
}

bool FrontierFinder::isNearUnknown(const Eigen::Vector3d& pos) {
  const int vox_num = floor(min_candidate_clearance_ / resolution_);
  for (int x = -vox_num; x <= vox_num; ++x)
    for (int y = -vox_num; y <= vox_num; ++y)
      for (int z = -1; z <= 1; ++z) {
        Eigen::Vector3d vox;
        vox << pos[0] + x * resolution_, pos[1] + y * resolution_, pos[2] + z * resolution_;
        if (edt_env_->sdf_map_->getOccupancy(vox) == SDFMap::UNKNOWN) return true;
      }
  return false;
}

int FrontierFinder::countVisibleCells(
    const Eigen::Vector3d& pos, const double& yaw, const vector<Eigen::Vector3d>& cluster) {
  percep_utils_->setPose(pos, yaw);
  int visib_num = 0;
  Eigen::Vector3i idx;
  for (auto cell : cluster) {
    // Check if frontier cell is inside FOV
    if (!percep_utils_->insideFOV(cell)) continue;

    // Check if frontier cell is visible (not occulded by obstacles)
    raycaster_->input(cell, pos);
    bool visib = true;
    while (raycaster_->nextId(idx)) {
      if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 ||
          edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
        visib = false;
        break;
      }
    }
    if (visib) visib_num += 1;
  }
  return visib_num;
}

void FrontierFinder::downsample(
    const vector<Eigen::Vector3d>& cluster_in, vector<Eigen::Vector3d>& cluster_out) {
  // downsamping cluster
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudf(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto cell : cluster_in)
    cloud->points.emplace_back(cell[0], cell[1], cell[2]);

  const double leaf_size = edt_env_->sdf_map_->getResolution() * down_sample_;
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(leaf_size, leaf_size, leaf_size);
  sor.filter(*cloudf);

  cluster_out.clear();
  for (auto pt : cloudf->points)
    cluster_out.emplace_back(pt.x, pt.y, pt.z);
}

void FrontierFinder::wrapYaw(double& yaw) {
  while (yaw < -M_PI)
    yaw += 2 * M_PI;
  while (yaw > M_PI)
    yaw -= 2 * M_PI;
}

Eigen::Vector3i FrontierFinder::searchClearVoxel(const Eigen::Vector3i& pt) {
  queue<Eigen::Vector3i> init_que;
  vector<Eigen::Vector3i> nbrs;
  Eigen::Vector3i cur, start_idx;
  init_que.push(pt);
  // visited_flag_[toadr(pt)] = 1;

  while (!init_que.empty()) {
    cur = init_que.front();
    init_que.pop();
    if (knownFree(cur)) {
      start_idx = cur;
      break;
    }

    nbrs = sixNeighbors(cur);
    for (auto nbr : nbrs) {
      int adr = toadr(nbr);
      // if (visited_flag_[adr] == 0)
      // {
      //   init_que.push(nbr);
      //   visited_flag_[adr] = 1;
      // }
    }
  }
  return start_idx;
}

bool FrontierFinder::satisfyFrontierCell(const Eigen::Vector3i& idx)
{
  if (knownFree(idx) && isNeighborUnknown(idx))
    return true;
  return false;
}

bool FrontierFinder::satisfySurfaceFrontierCell(const Eigen::Vector3i& idx)
{
  if (knownFree(idx) && isNeighborUnknown(idx) && isNeighborOccupied(idx))
    return true;
  return false;
}

inline vector<Eigen::Vector3i> FrontierFinder::sixNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(6);
  Eigen::Vector3i tmp;

  tmp = voxel - Eigen::Vector3i(1, 0, 0);
  neighbors[0] = tmp;
  tmp = voxel + Eigen::Vector3i(1, 0, 0);
  neighbors[1] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 1, 0);
  neighbors[2] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 1, 0);
  neighbors[3] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 0, 1);
  neighbors[4] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 0, 1);
  neighbors[5] = tmp;

  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::tenNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(10);
  Eigen::Vector3i tmp;
  int count = 0;

  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      if (x == 0 && y == 0) continue;
      tmp = voxel + Eigen::Vector3i(x, y, 0);
      neighbors[count++] = tmp;
    }
  }
  neighbors[count++] = tmp - Eigen::Vector3i(0, 0, 1);
  neighbors[count++] = tmp + Eigen::Vector3i(0, 0, 1);
  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::allNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(26);
  Eigen::Vector3i tmp;
  int count = 0;
  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z) {
        if (x == 0 && y == 0 && z == 0) continue;
        tmp = voxel + Eigen::Vector3i(x, y, z);
        neighbors[count++] = tmp;
      }
  return neighbors;
}

inline bool FrontierFinder::isNeighborUnknown(const Eigen::Vector3i& voxel)
{
  // At least one neighbor is unknown
  auto nbrs = sixNeighbors(voxel);
  for (auto nbr : nbrs) {
    if (edt_env_->sdf_map_->isInBox(nbr) &&
        edt_env_->sdf_map_->getOccupancy(nbr) == SDFMap::UNKNOWN)
      return true;
  }
  return false;
}

inline bool FrontierFinder::isNeighborOccupied(const Eigen::Vector3i& voxel)
{
  // At least one neighbor is occupied
  auto nbrs = eighteenNeighbors(voxel);
  for (auto nbr : nbrs) {
    if (edt_env_->sdf_map_->isInBox(nbr) &&
        edt_env_->sdf_map_->getOccupancy(nbr) == SDFMap::OCCUPIED)
      return true;
  }
  return false;
}

inline vector<Eigen::Vector3i> FrontierFinder::eighteenNeighbors(const Eigen::Vector3i& voxel)
{
  vector<Eigen::Vector3i> neighbors(18);
  Eigen::Vector3i tmp;
  int count = 0;
  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z) {
        if (x == 0 && y == 0 && z == 0)
          continue;
        if (abs(x) + abs(y) + abs(z) > 2)
          continue;
        tmp = voxel + Eigen::Vector3i(x, y, z);
        neighbors[count++] = tmp;
      }
  return neighbors;
}

inline int FrontierFinder::toadr(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->toAddress(idx);
}

inline bool FrontierFinder::knownFree(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::FREE;
}

inline bool FrontierFinder::inmap(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->isInMap(idx);
}

}  // namespace fast_planner