
#include <cmath>
#include <string>
#include "RecoHGCal/TICL/plugins/SmoothingAlgoByMLP.h"

#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/HGCalReco/interface/Common.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"

#include <bits/stdc++.h>

using namespace std;
using namespace ticl;
using namespace cms::Ort;


SmoothingAlgoByMLP::SmoothingAlgoByMLP(const edm::ParameterSet &conf)
    : LinkingAlgoBase(conf),
      del_tk_ts_layer1_(conf.getParameter<double>("delta_tk_ts_layer1")),
      del_tk_ts_int_(conf.getParameter<double>("delta_tk_ts_interface")),
      del_ts_em_had_(conf.getParameter<double>("delta_ts_em_had")),
      del_ts_had_had_(conf.getParameter<double>("delta_ts_had_had")),
      timing_quality_threshold_(conf.getParameter<double>("track_time_quality_threshold")),
      cutTk_(conf.getParameter<std::string>("cutTk")) {}

SmoothingAlgoByMLP::~SmoothingAlgoByMLP() {}

void SmoothingAlgoByMLP::initialize(
    const HGCalDDDConstants *hgcons,
    const hgcal::RecHitTools rhtools,
    const edm::ESHandle<MagneticField> bfieldH,
    const edm::ESHandle<Propagator> propH)
{
  hgcons_ = hgcons;
  rhtools_ = rhtools;

  bfield_ = bfieldH;
  propagator_ = propH;
}


/** MLP FEATURE MAP

  // main trackster
  0 "barycenter_x",
  1 "barycenter_y",
  2 "barycenter_z",
  3 "raw_energy",
  4 "raw_em_energy",
  5 "EV1",
  6 "EV2",
  7 "EV3",
  8 "eVector0_x",
  9 "eVector0_y",
  10 "eVector0_z",
  11 "sigmaPCA1",
  12 "sigmaPCA2",
  13 "sigmaPCA3",

  // candidate trackster
  14 "barycenter_x",
  15 "barycenter_y",
  16 "barycenter_z",
  17 "raw_energy",
  18 "raw_em_energy",
  19 "EV1",
  20 "EV2",
  21 "EV3",
  22 "eVector0_x",
  23 "eVector0_y",
  24 "eVector0_z",
  25 "sigmaPCA1",
  26 "sigmaPCA2",
  27 "sigmaPCA3",

  // main trackster
  28 min_z_point_x,
  29 min_z_point_y,
  30 min_z_point_z,
  31 max_z_point_x,
  32 max_z_point_y,
  33 max_z_point_z,

  // candidate trackster
  34 min_z_point_x,
  35 min_z_point_y,
  36 min_z_point_z,
  37 max_z_point_x,
  38 max_z_point_y,
  39 max_z_point_z,

  // shared
  40 min_pairwise_planear_distance,

  // main trackster
  41 num_lc

  // candidate trackster
  42 num_lc
*/

void SmoothingAlgoByMLP::SmoothingAlgoByMLP(
    const edm::Handle<std::vector<reco::Track>> tkH,
    const edm::ValueMap<float> &tkTime,
    const edm::ValueMap<float> &tkTimeErr,
    const edm::ValueMap<float> &tkTimeQual,
    const std::vector<reco::Muon> &muons,
    const edm::Handle<std::vector<Trackster>> tsH,
    std::vector<TICLCandidate> &resultLinked,
    std::vector<TICLCandidate> &chargedHadronsFromTk,
    const TICLGraph &ticlGraph, // do not need this
    const ONNXRuntime *cache)
{
  std::cout << "Smoothing Algo by MLP " << std::endl;
  const auto &tracks = *tkH;
  const auto &tracksters = *tsH;

  auto bFieldProd = bfield_.product();
  const Propagator &prop = (*propagator_);

  /** PREPARING FEATURES **/

  const std::vector<std::string> input_names = {"features"};

  long int N = tracksters.size();
  const auto shapeFeatures = 43;

  FloatArrays data;
  std::vector<std::vector<int64_t>> input_shapes;

  std::vector<float> features;

  // Assuming this method is called per event
  // steps:
  // 2. get_trackster_representative_points (min z-point, max z-point)
  //  - idea simplify the cone creation, connect min and max-z points to create the cone

  // Print out info about tracksters
  std::cout << "Number of tracksters in event: " << N << std::endl;
  for (unsigned i = 0; i < tracksters.size(); ++i) {
    const auto &ts = tracksters[i];
    const float raw_energy = ts.raw_energy();

    // ignore low energy tracksters
    if (raw_energy < 50) {
      continue;
    }


    const Vector &barycenter = ts.barycenter();
    const Vector &eigenvector0 = ts.eigenvectors(0);
    const std::array<float, 3> &eigenvalues = ts.eigenvalues();
    const std::array<float, 3> &sigmasPCA = ts.sigmasPCA();


    // 1. we got a major trackster we want to smooth
    std::cout << "Trackster " << i << "--------------------" << std::endl;
    std::cout << "Raw Energy: " << raw_energy << std::endl;

    // 2. get representative points of the trackster (where (0, 0, 0) -> (bx, by, bz) intersects the min and max layer)
    // min_z = min(vertices_z[bigT])
    // max_z = max(vertices_z[bigT])
    // ts.vertices() are indexes to global collection - need to get the global collection retrieve the vertices
    // and then find the minimum

    // assume we got this for now
    const float min_z;
    const float max_z;
    const float bx = barycenter.x();
    const float by = barycenter.y();
    const float bz = barycenter.z();

    /*
      Representative points of the cone (alternatively use min_z, max_z)
      t_min = min_z / bz
      t_max = max_z / bz
      x1 = np.array((t_min * bx, t_min * by, min_z))
      x2 = np.array((t_max * bx, t_max * by, max_z))
      return x1, x2
    */

    // Loop over tracksters and see if they are in the cone
    for (unsigned ci = 0; ci < tracksters.size(); ++ci) {

      // no self loops
      if (ci == i) {
        continue;
      }

      // candidate trackster
      const auto &ct = tracksters[ci];

      /*
        in_cone = []
        for i, x0 in enumerate(barycentres):
            # barycenter between the first and last layer
            if x0[2] > x1[2] - radius and x0[2] < x2[2] + radius:
                # distance from the particle axis less than X cm
                d = np.linalg.norm(np.cross(x0 - x1, x0 - x2)) / np.linalg.norm(x2 - x1)
                if d < radius:
                    in_cone.append((i, d))
        return in_cone
      */

      features.push_back(barycenter.x());     // 0
      features.push_back(barycenter.y());     // 1
      features.push_back(barycenter.z());     // 2
      features.push_back(raw_energy);         // 3
      features.push_back(ts.raw_em_energy()); // 4
      features.push_back(eigenvalues[0]);     // 5
      features.push_back(eigenvalues[1]);     // 6
      features.push_back(eigenvalues[2]);     // 7
      features.push_back(eigenvector0.x());   // 8
      features.push_back(eigenvector0.y());   // 9
      features.push_back(eigenvector0.z());   // 10
      features.push_back(sigmasPCA[0]);       // 11
      features.push_back(sigmasPCA[1]);       // 12
      features.push_back(sigmasPCA[2]);       // 13

      const Vector &c_barycenter = ct.barycenter();
      const Vector &c_eigenvector0 = ct.eigenvectors(0);
      const std::array<float, 3> &c_eigenvalues = ct.eigenvalues();
      const std::array<float, 3> &_sigmasPCA = ct.sigmasPCA();

      features.push_back(c_barycenter.x());     // 14
      features.push_back(c_barycenter.y());     // 15
      features.push_back(c_barycenter.z());     // 16
      features.push_back(ct.raw_energy());      // 17
      features.push_back(ct.raw_em_energy());   // 18
      features.push_back(c_eigenvalues[0]);     // 19
      features.push_back(c_eigenvalues[1]);     // 20
      features.push_back(c_eigenvalues[2]);     // 21
      features.push_back(c_eigenvector0.x());   // 22
      features.push_back(c_eigenvector0.y());   // 23
      features.push_back(c_eigenvector0.z());   // 24
      features.push_back(c_sigmasPCA[0]);       // 25
      features.push_back(c_sigmasPCA[1]);       // 26
      features.push_back(c_sigmasPCA[2]);       // 27

      // 28 min_z_point_x,
      // 29 min_z_point_y,
      // 30 min_z_point_z,
      // 31 max_z_point_x,
      // 32 max_z_point_y,
      // 33 max_z_point_z,

      // // candidate trackster
      // 34 min_z_point_x,
      // 35 min_z_point_y,
      // 36 min_z_point_z,
      // 37 max_z_point_x,
      // 38 max_z_point_y,
      // 39 max_z_point_z,

      // // shared
      // 40 min_pairwise_planear_distance,

      features.push_back(ts.vertices().size()); // 41
      features.push_back(ct.vertices().size()); // 42
    }


    // keep track of the shape
    // candidate labels


    std::cout << "--------------------" << std::endl;
  }

  /** RUNNING THE NETWORK **/

  // Get network output
  std::vector<float> edge_predictions = cache->run(input_names, data, input_shapes)[0];

  for (int i = 0; i < static_cast<int>(edge_predictions.size()); i++) {
    std::cout << "Network output for edge " << data[1][i] << "-" << data[1][numEdges + i]
              << " is: " << edge_predictions[i] << std::endl;
  }

  // Create a graph
  Graph g;
  const auto classification_threshold = 0.7;

  // Self-loop for not connected nodes
  for (int i = 0; i < N; i++){
    g.addEdge(i, i);
  }
  // Building a predicted graph
  for (int i = 0; i < numEdges; i++) {
    if (edge_predictions[i] >= classification_threshold) {
      auto src = data[1][i];
      auto dst = data[1][numEdges + i];
      // Make undirectional
      g.addEdge(src, dst);
      g.addEdge(dst, src);
    }
  }

  std::cout << "Following is Depth First Traversal" << std::endl;
  std::cout << "Connected components are: " << std::endl;
  g.DFS();

  int i = 0;
  std::vector<TICLCandidate> connectedCandidates;

  for (auto &component : g.connected_components) {
    TICLCandidate tracksterCandidate;
    for (auto &trackster_id : component) {
      std::cout << "Component " << i << ": trackster id " << trackster_id << std::endl;
      tracksterCandidate.addTrackster(edm::Ptr<Trackster>(tsH, trackster_id));
    }
    i++;
    connectedCandidates.push_back(tracksterCandidate);
  }

  // The final candidates are passed to `resultLinked`
  resultLinked.insert(std::end(resultLinked), std::begin(connectedCandidates), std::end(connectedCandidates));

}  // linkTracksters


void LinkingAlgoByGNN::fillPSetDescription(edm::ParameterSetDescription &desc) {
  desc.add<std::string>("cutTk",
                        "1.48 < abs(eta) < 3.0 && pt > 1. && quality(\"highPurity\") && "
                        "hitPattern().numberOfLostHits(\"MISSING_OUTER_HITS\") < 5");
  desc.add<double>("delta_tk_ts_layer1", 0.02);
  desc.add<double>("delta_tk_ts_interface", 0.03);
  desc.add<double>("delta_ts_em_had", 0.03);
  desc.add<double>("delta_ts_had_had", 0.03);
  desc.add<double>("track_time_quality_threshold", 0.5);
  LinkingAlgoBase::fillPSetDescription(desc);
}
