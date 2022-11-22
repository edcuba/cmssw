
#include <cmath>
#include <string>
#include "RecoHGCal/TICL/plugins/LinkingAlgoByGNN.h"

#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/HGCalReco/interface/Common.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"

#include <bits/stdc++.h>

using namespace std;
using namespace ticl;
using namespace cms::Ort;

class Graph {
  // A function used by DFS
  void DFSUtil(int v);

public:
  map<int, bool> visited;
  map<int, list<int>> adj;
  std::vector<std::vector<int>> connected_components;
  // function to add an edge to graph
  void addEdge(int v, int w);

  // prints DFS traversal of the complete graph
  void DFS();
};

void Graph::addEdge(int v, int w) {
  adj[v].push_back(w);  // Add w to v�s list.
}

void Graph::DFSUtil(int v) {
  // Mark the current node as visited and print it
  visited[v] = true;
  //std::cout << v << std::endl;

  // Recur for all the vertices adjacent to this vertex
  list<int>::iterator i;
  for (auto i = adj[v].begin(); i != adj[v].end(); ++i)
    if (!visited[*i]) {
      connected_components.back().push_back(*i);
      std::cout << "Pushed back " << *i << std::endl;
      DFSUtil(*i);
    }
}

// The function to do DFS traversal. It uses recursive DFSUtil()
void Graph::DFS() {
  // Call the recursive helper function to print DFS
  // traversal starting from all vertices one by one
  for (auto i : adj)
    if (visited[i.first] == false) {
      std::cout << "Emplaced back: " << i.first << std::endl;
      connected_components.emplace_back(1, i.first);
      std::cout << "Starting DFS from node: " << i.first << std::endl;
      DFSUtil(i.first);
    }
}

LinkingAlgoByGNN::LinkingAlgoByGNN(const edm::ParameterSet &conf)
    : LinkingAlgoBase(conf),
      del_tk_ts_layer1_(conf.getParameter<double>("delta_tk_ts_layer1")),
      del_tk_ts_int_(conf.getParameter<double>("delta_tk_ts_interface")),
      del_ts_em_had_(conf.getParameter<double>("delta_ts_em_had")),
      del_ts_had_had_(conf.getParameter<double>("delta_ts_had_had")),
      timing_quality_threshold_(conf.getParameter<double>("track_time_quality_threshold")),
      cutTk_(conf.getParameter<std::string>("cutTk")) {}

LinkingAlgoByGNN::~LinkingAlgoByGNN() {}

void LinkingAlgoByGNN::initialize(const HGCalDDDConstants *hgcons,
                                  const hgcal::RecHitTools rhtools,
                                  const edm::ESHandle<MagneticField> bfieldH,
                                  const edm::ESHandle<Propagator> propH) {
  hgcons_ = hgcons;
  rhtools_ = rhtools;

  bfield_ = bfieldH;
  propagator_ = propH;
}

void LinkingAlgoByGNN::linkTracksters(const edm::Handle<std::vector<reco::Track>> tkH,
                                      const edm::ValueMap<float> &tkTime,
                                      const edm::ValueMap<float> &tkTimeErr,
                                      const edm::ValueMap<float> &tkTimeQual,
                                      const std::vector<reco::Muon> &muons,
                                      const edm::Handle<std::vector<Trackster>> tsH,
                                      std::vector<TICLCandidate> &resultLinked,
                                      std::vector<TICLCandidate> &chargedHadronsFromTk,
                                      const TICLGraph &ticlGraph,
                                      const ONNXRuntime *cache) {
                                      
  std::cout << "Linking Algo by GNN " << std::endl;
  const auto &tracks = *tkH;
  const auto &tracksters = *tsH;

  auto bFieldProd = bfield_.product();
  const Propagator &prop = (*propagator_);

  const std::vector<std::string> input_names = {"features", "edge_index", "adj", "trackster_index"};

  long int N = tracksters.size();
  const auto shapeFeatures = 15;

  FloatArrays data;
  std::vector<std::vector<int64_t>> input_shapes;

  std::vector<float> features;
  // For nearest neighbour finding
  std::vector<float> barycenters_x;
  std::vector<float> barycenters_y;
  std::vector<float> barycenters_z;
  

  // Print out info about tracksters
  std::cout << "Number of tracksters in event: " << N << std::endl;
  for (unsigned i = 0; i < tracksters.size(); ++i) {
    const auto &ts = tracksters[i];
    std::cout << "Trackster " << i << "--------------------" << std::endl;
    std::cout << "Barycenter X: " << ts.barycenter().x() << std::endl;
    std::cout << "Barycenter Y: " << ts.barycenter().y() << std::endl;
    std::cout << "Barycenter Z: " << ts.barycenter().z() << std::endl;

    Vector eigenvector0 = ts.eigenvectors(0);
    std::cout << "eVector0 X: " << eigenvector0.x() << std::endl;
    std::cout << "eVector0 Y: " << eigenvector0.y() << std::endl;
    std::cout << "eVector0 Z: " << eigenvector0.z() << std::endl;

    std::array<float, 3> eigenvalues = ts.eigenvalues();
    std::cout << "EV1: " << eigenvalues[0] << std::endl;
    std::cout << "EV2: " << eigenvalues[1] << std::endl;
    std::cout << "EV3: " << eigenvalues[2] << std::endl;

    std::array<float, 3> sigmasPCA = ts.sigmasPCA();
    std::cout << "sigmaPCA1: " << sigmasPCA[0] << std::endl;
    std::cout << "sigmaPCA2: " << sigmasPCA[1] << std::endl;
    std::cout << "sigmaPCA3: " << sigmasPCA[2] << std::endl;

    // size
    std::cout << "Size: " << ts.vertices().size() << std::endl;
    std::cout << "Raw Energy: " << ts.raw_energy() << std::endl;
    std::cout << "Raw EM Energy: " << ts.raw_em_energy() << std::endl;

    // candidate labels
    /* Python:
    
    in_candidate = [-1 for i in range(trk_data[ev].NTracksters)]
            for indx, cand in enumerate(cand_data[ev].tracksters_in_candidate):
                for ts in cand:
                    in_candidate[ts] = indx
    */
    features.push_back(ts.barycenter().x());
    features.push_back(ts.barycenter().y());
    features.push_back(ts.barycenter().z());    
    features.push_back(eigenvector0.x());
    features.push_back(eigenvector0.y());
    features.push_back(eigenvector0.z());
    features.push_back(eigenvalues[0]);
    features.push_back(eigenvalues[1]);
    features.push_back(eigenvalues[2]);
    features.push_back(sigmasPCA[0]);
    features.push_back(sigmasPCA[1]);
    features.push_back(sigmasPCA[2]);
    features.push_back(ts.vertices().size());
    features.push_back(ts.raw_energy());
    features.push_back(ts.raw_em_energy());
    
    barycenters_x.push_back(ts.barycenter().x());
    barycenters_y.push_back(ts.barycenter().y());
    barycenters_z.push_back(ts.barycenter().z());
    

    std::cout << "--------------------" << std::endl;
  }

  input_shapes.push_back({1, N, shapeFeatures});
  data.emplace_back(features);

  // Creating Edges: uncomment when have a Graph as an input

  std::vector<float> edges_src;
  std::vector<float> edges_dst;
  for (int i = 0; i < N; i++){
    for (auto & i_neighbour : ticlGraph.getNode(i).getInner()){
      // Create an edge between the tracksters
      edges_src.push_back(static_cast<float>(i_neighbour));
      edges_dst.push_back(static_cast<float>(i));
    }
    
    /* 
    // Nearest Neighbour connection
    if (len(ticlGraph.getNode(i).getInner()) == 0 && len(ticlGraph.getNode(i).getOuter()) == 0){
     
      const auto pos_i = [barycenters_x[i], barycenters_y[i], barycenters_z[i]];
      const auto d_least = 1000;
      for (auto k=0; k< len(barycenters_x; k++){
          if (k == i){
              continue;
          }
          pos_k = [barycenters_x[k], barycenters_y[k], barycenters_z[k]];
          del_pos = pos_k - pos_i;
          d_squared = del_pos[0]**2 + del_pos[1]**2 + del_pos[2]**2;
          if (d_squared < d_least){
              d_least = d_squared;
              i_least = k;
          }
      }
              
      const auto nearest_id = findNearestNeighbour(i, b_x, b_y, b_z)
      edges_src.push_back(static_cast<float>(nearest_id));
      edges_dst.push_back(static_cast<float>(i));
      }
    */
  }

  auto numEdges = static_cast<int>(edges_src.size());
  
  // Do not run the network in these cases
  if (numEdges == 0 || N < 2){
  
    std::vector<TICLCandidate> connectedCandidates;
  
    for (int trackster_id = 0; trackster_id < N; trackster_id++) {
    
      TICLCandidate tracksterCandidate;
      tracksterCandidate.addTrackster(edm::Ptr<Trackster>(tsH, trackster_id));
      connectedCandidates.push_back(tracksterCandidate);
    }
  
    // The final candidates are passed to `resultLinked`
    resultLinked.insert(std::end(resultLinked), std::begin(connectedCandidates), std::end(connectedCandidates));
    return;
  }
  
  input_shapes.push_back({1, 2, numEdges});
  std::cout << "Num edges: " << numEdges << std::endl;

  data.emplace_back(edges_src);
  for (auto &dst : edges_dst) {
    data.back().push_back(dst);
  }

  
  // Creating Adjacency matrix and trackster index
  std::vector<float> A;
  std::vector<float> trackster_index;
  
  // Add self-loops
  /*
  for (int i=0; i<N; i++){
    for (int j=0; j<N; j++){
      if (i == j){
        A.push_back(1.);
        trackster_index.push_back(1.);
      }
      else {
        A.push_back(0.);
        trackster_index.push_back(0.);
      }
    }
  }
  */
  
  // Add undirectional edges
  for (int i=0; i<numEdges; i++){
    // for source
    for (int j=0; j<N; j++){
      if (j == edges_src[i]){A.push_back(1.);} //static_cast<float>(1.0)
      else {A.push_back(0.);}
      if (j == edges_dst[i]){trackster_index.push_back(1.);}
      else {trackster_index.push_back(0.);}
    }
    
    // for dst
    for (int j=0; j<N; j++){
      if (j == edges_dst[i]){A.push_back(1.);}
      else {A.push_back(0.);}
      if (j == edges_src[i]){trackster_index.push_back(1.);}
      else {trackster_index.push_back(0.);}
    }
  }
  
  //input_shapes.push_back({1, 2*numEdges+N, N});
  input_shapes.push_back({1, 2*numEdges, N});
  data.emplace_back(A);
  //input_shapes.push_back({1, 2*numEdges+N, N});
  input_shapes.push_back({1, 2*numEdges, N});
  data.emplace_back(trackster_index);
  
  std::cout << "Adj size: " << A.size() << std::endl;

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
