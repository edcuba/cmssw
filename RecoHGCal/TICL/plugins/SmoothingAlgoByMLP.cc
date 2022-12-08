
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

void SmoothingAlgoByMLP::linkTracksters(
    const edm::Handle<std::vector<reco::Track>> tkH,
    const edm::ValueMap<float> &tkTime,
    const edm::ValueMap<float> &tkTimeErr,
    const edm::ValueMap<float> &tkTimeQual,
    const std::vector<reco::Muon> &muons,
    const edm::Handle<std::vector<Trackster>> tsH,
    std::vector<TICLCandidate> &resultLinked,
    std::vector<TICLCandidate> &chargedHadronsFromTk,
    const TICLGraph &ticlGraph,
    const std::vector<reco::CaloCluster>& layerClusters,
    const ONNXRuntime *cache)
{
  std::cout << "Smoothing Algo by MLP " << std::endl;

  const auto &tracksters = *tsH;

  const float classification_threshold = 0.7;
  const float radius = 10;

  /** PREPARING FEATURES **/

  const std::vector<std::string> input_names = {"features"};

  long int N = tracksters.size();
  const auto shapeFeatures = 43;

  std::vector<float> features;
  std::vector<unsigned> big_tracksters;
  std::vector<std::pair<unsigned, unsigned>> pairs;

  // Assuming this method is called per event
  // steps:
  // 2. get_trackster_representative_points (min z-point, max z-point)

  std::cout << "Number of tracksters in event: " << N << std::endl;
  for (unsigned i = 0; i < tracksters.size(); ++i) {
    const auto &ts = tracksters[i];
    const float raw_energy = ts.raw_energy();

    // 1. we got a major trackster we want to smooth
    // ignore low energy tracksters
    if (raw_energy < 50) {
      continue;
    }

    const Vector &barycenter = ts.barycenter();
    const Vector &eigenvector0 = ts.eigenvectors(0);
    const std::array<float, 3> &eigenvalues = ts.eigenvalues();
    const std::array<float, 3> &sigmasPCA = ts.sigmasPCA();

    // 2. get representative points of the trackster (where (0, 0, 0) -> (bx, by, bz) intersects the min and max layer)
    const std::vector<unsigned int> &vertices_indices = ts.vertices();

    const auto max_z_lc_it = std::max_element(
      vertices_indices.begin(),
      vertices_indices.end(),
      [&layerClusters](const int &a, const int &b) {
        return layerClusters[a].z() > layerClusters[b].z();
      }
    );

    const auto min_z_lc_it = std::min_element(
      vertices_indices.begin(),
      vertices_indices.end(),
      [&layerClusters](const int &a, const int &b) {
        return layerClusters[a].z() > layerClusters[b].z();
      }
    );

    const reco::CaloCluster &min_z_lc = layerClusters[*min_z_lc_it];
    const reco::CaloCluster &max_z_lc = layerClusters[*max_z_lc_it];

    // compute the cylinder bounds
    const float t_min = min_z_lc.z() / barycenter.z();
    const float t_max = max_z_lc.z() / barycenter.z();

    const Vector x1 = Vector(
      t_min * barycenter.x(),
      t_min * barycenter.y(),
      min_z_lc.z()
    );

    const Vector x2 = Vector(
      t_max * barycenter.x(),
      t_max * barycenter.y(),
      min_z_lc.z()
    );

    // Loop over tracksters and see if they are in the cone
    for (unsigned ci = 0; ci < tracksters.size(); ++ci) {

      // no self loops
      if (ci == i) {
        continue;
      }

      // candidate trackster
      const auto &ct = tracksters[ci];

      const Vector &c_barycenter = ct.barycenter();
      const Vector &c_eigenvector0 = ct.eigenvectors(0);
      const std::array<float, 3> &c_eigenvalues = ct.eigenvalues();
      const std::array<float, 3> &c_sigmasPCA = ct.sigmasPCA();

      const std::vector<unsigned int> &c_vertices_indices = ct.vertices();

      const auto c_max_z_lc_it = std::max_element(
        c_vertices_indices.begin(),
        c_vertices_indices.end(),
        [&layerClusters](const int &a, const int &b) {
          return layerClusters[a].z() > layerClusters[b].z();
        }
      );

      const auto c_min_z_lc_it = std::min_element(
        c_vertices_indices.begin(),
        c_vertices_indices.end(),
        [&layerClusters](const int &a, const int &b) {
          return layerClusters[a].z() > layerClusters[b].z();
        }
      );

      const reco::CaloCluster &c_min_z_lc = layerClusters[*c_min_z_lc_it];
      const reco::CaloCluster &c_max_z_lc = layerClusters[*c_max_z_lc_it];

      // compute the distance and position in the cone
      if (c_barycenter.z() < x1.z() - radius || c_barycenter.z() > x2.z() + radius) {
        // not in z-cone bounds
        continue;
      }

      // d = np.linalg.norm(np.cross(x0 - x1, x0 - x2)) / np.linalg.norm(x2 - x1)
      const float distance = (c_barycenter - x1).Cross(c_barycenter - x2).R() / (x2 - x1).R();

      if (distance > radius) {
        // not in cone
        continue;
      }

      features.insert(features.end(), {
        (float) barycenter.x(),     // 0
        (float) barycenter.y(),     // 1
        (float) barycenter.z(),     // 2
        (float) raw_energy,         // 3
        (float) ts.raw_em_energy(), // 4
        (float) eigenvalues[0],     // 5
        (float) eigenvalues[1],     // 6
        (float) eigenvalues[2],     // 7
        (float) eigenvector0.x(),   // 8
        (float) eigenvector0.y(),   // 9
        (float) eigenvector0.z(),   // 10
        (float) sigmasPCA[0],       // 11
        (float) sigmasPCA[1],       // 12
        (float) sigmasPCA[2],       // 13
        (float) c_barycenter.x(),   // 14
        (float) c_barycenter.y(),   // 15
        (float) c_barycenter.z(),   // 16
        (float) ct.raw_energy(),    // 17
        (float) ct.raw_em_energy(), // 18
        (float) c_eigenvalues[0],   // 19
        (float) c_eigenvalues[1],   // 20
        (float) c_eigenvalues[2],   // 21
        (float) c_eigenvector0.x(), // 22
        (float) c_eigenvector0.y(), // 23
        (float) c_eigenvector0.z(), // 24
        (float) c_sigmasPCA[0],     // 25
        (float) c_sigmasPCA[1],     // 26
        (float) c_sigmasPCA[2],     // 27
        (float) min_z_lc.x(),       // 28
        (float) min_z_lc.y(),       // 29
        (float) min_z_lc.z(),       // 30
        (float) max_z_lc.x(),       // 31
        (float) max_z_lc.y(),       // 32
        (float) max_z_lc.z(),       // 33
        (float) c_min_z_lc.x(),     // 34
        (float) c_min_z_lc.y(),     // 35
        (float) c_min_z_lc.z(),     // 36
        (float) c_max_z_lc.x(),     // 37
        (float) c_max_z_lc.y(),     // 38
        (float) c_max_z_lc.z(),     // 39
        (float) distance,           // 40
        (float) ts.vertices().size(), // 41
        (float) ct.vertices().size() // 42
      });

      // keep track of the samples
      pairs.push_back(std::make_pair(i, ci));

      if (big_tracksters.back() != i) {
        // keep tracks of big tracksters
        big_tracksters.push_back(i);
      }
    }
  }

  /** RUNNING THE NETWORK **/

  // Prepare network input
  std::vector<std::vector<int64_t>> input_shapes;
  input_shapes.push_back({ (int64_t) pairs.size(), shapeFeatures });

  FloatArrays data;
  data.emplace_back(features);

  // get the network output
  std::vector<float> edge_predictions = cache->run(input_names, data, input_shapes)[0];


  // interpret the results
  std::vector<TICLCandidate> connectedCandidates;

  for (const int bt : big_tracksters) {

    TICLCandidate tracksterCandidate;
    tracksterCandidate.addTrackster(edm::Ptr<Trackster>(tsH, bt));

    int yi = 0;
    for (auto &p : pairs) {
      const int pi = p.first;
      const int pj = p.second;

      if (bt == pi && edge_predictions[yi++] > classification_threshold) {
        // check if the score is > threshold
        tracksterCandidate.addTrackster(edm::Ptr<Trackster>(tsH, pj));
      }
    }

    if (tracksterCandidate.tracksters().size() > 1) {
      // if anything to connect to
      connectedCandidates.push_back(tracksterCandidate);
    }
  }

  // The final candidates are passed to `resultLinked`
  resultLinked.insert(std::end(resultLinked), std::begin(connectedCandidates), std::end(connectedCandidates));

}  // linkTracksters


void SmoothingAlgoByMLP::fillPSetDescription(edm::ParameterSetDescription &desc) {
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
