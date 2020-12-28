#pragma once
#include "gpu_graph.cuh"
#include "sampler_result.cuh"
// #include "alias_table.cuh"
#include <random>
DECLARE_bool(ol);
DECLARE_bool(umtable);
DECLARE_bool(hmtable);
// struct sample_result;
// class Sampler;

template <typename T> void printH(T *ptr, int size) {
  T *ptrh = new T[size];
  H_ERR(cudaMemcpy(ptrh, ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
  printf("printH: ");
  for (size_t i = 0; i < size; i++) {
    // printf("%d\t", ptrh[i]);
    std::cout << ptrh[i] << "\t";
  }
  printf("\n");
  delete ptrh;
}

// template <JobType T = JobType::NS> class Sampler;

class Sampler {
public:
  gpu_graph ggraph;
  sample_result result;
  uint num_seed;
  // Jobs_result<JobType::RW, T> rw_result;

  float *prob_array;
  uint *alias_array;
  char *valid;
  // float *avg_bias;

public:
  Sampler(gpu_graph graph) {
    ggraph = graph;
    // Init();
  }
  ~Sampler() {}
  void AllocateAliasTable() {
    if (!FLAGS_umtable && !FLAGS_hmtable) {
      H_ERR(cudaMalloc((void **)&prob_array, ggraph.edge_num * sizeof(float)));
      H_ERR(cudaMalloc((void **)&alias_array, ggraph.edge_num * sizeof(uint)));
      H_ERR(cudaMalloc((void **)&valid, ggraph.vtx_num * sizeof(char)));
    }
    if (FLAGS_umtable) {
      H_ERR(cudaMallocManaged((void **)&prob_array,
                              ggraph.edge_num * sizeof(float)));
      H_ERR(cudaMallocManaged((void **)&alias_array,
                              ggraph.edge_num * sizeof(uint)));
      H_ERR(cudaMallocManaged((void **)&valid, ggraph.vtx_num * sizeof(char)));

      H_ERR(cudaMemAdvise(prob_array, ggraph.edge_num * sizeof(float),
                          cudaMemAdviseSetAccessedBy, FLAGS_device));
      H_ERR(cudaMemAdvise(alias_array, ggraph.edge_num * sizeof(uint),
                          cudaMemAdviseSetAccessedBy, FLAGS_device));
      H_ERR(cudaMemAdvise(valid, ggraph.vtx_num * sizeof(char),
                          cudaMemAdviseSetAccessedBy, FLAGS_device));
    }
    if (FLAGS_hmtable) {
      if(FLAGS_v)
        printf("host mapped table");
      H_ERR(cudaHostAlloc((void **)&prob_array, ggraph.edge_num * sizeof(float),
                          cudaHostAllocWriteCombined ));
      H_ERR(cudaHostAlloc((void **)&alias_array, ggraph.edge_num * sizeof(uint),
                          cudaHostAllocWriteCombined ));
      H_ERR(cudaHostAlloc((void **)&valid, ggraph.vtx_num * sizeof(char),
                          cudaHostAllocWriteCombined ));
    }
    // if (!FLAGS_ol)
    //   H_ERR(cudaMalloc((void **)&avg_bias, ggraph.vtx_num * sizeof(float)));
    ggraph.valid = valid;
    ggraph.prob_array = prob_array;
    ggraph.alias_array = alias_array;
    H_ERR(cudaMemset(prob_array, 0, ggraph.vtx_num * sizeof(float)));
  }
  void SetSeed(uint _num_seed, uint _hop_num, uint *_hops) {
    // printf("%s\t %s :%d\n", __FILE__, __PRETTY_FUNCTION__, __LINE__);
    num_seed = _num_seed;
    std::random_device rd;
    std::mt19937 gen(56);
    std::uniform_int_distribution<> dis(1, 10000); // ggraph.vtx_num);
    uint *seeds = new uint[num_seed];
    for (int n = 0; n < num_seed; ++n) {
#ifdef check
      // seeds[n] = n;
      seeds[n] = 1;
// seeds[n] = 339;
#else
      seeds[n] = n;
// seeds[n] = dis(gen);
#endif // check
    }
    result.init(num_seed, _hop_num, _hops, seeds);
    // printf("first ten seed:");
    // printH(result.data,10 );
  }
  void InitFullForConstruction() {
    uint *seeds = new uint[ggraph.vtx_num];
    for (int n = 0; n < ggraph.vtx_num; ++n) {
      seeds[n] = n;
    }
    // uint _hops[2] = {1, 1};
    uint *_hops = new uint[2];
    _hops[0] = 1;
    _hops[1] = 1;
    result.init(ggraph.vtx_num, 2, _hops, seeds);
  }
  // void Start();
};

class Walker {
public:
  gpu_graph ggraph;
  // sample_result result;
  uint num_seed;
  Jobs_result<JobType::RW, uint> result;

  float *prob_array;
  uint *alias_array;
  char *valid;

public:
  Walker(gpu_graph graph) {
    ggraph = graph;
    // Init();
  }
  Walker(Sampler &sampler) {
    ggraph = sampler.ggraph;
    valid = ggraph.valid;
    prob_array = ggraph.prob_array;
    alias_array = ggraph.alias_array;
  }
  ~Walker() {}
  __device__ void BindResult() { ggraph.result = &result; }
  void AllocateAliasTable() {
    H_ERR(cudaMalloc((void **)&prob_array, ggraph.edge_num * sizeof(float)));
    H_ERR(cudaMalloc((void **)&alias_array, ggraph.edge_num * sizeof(uint)));
    H_ERR(cudaMalloc((void **)&valid, ggraph.vtx_num * sizeof(char)));
    H_ERR(cudaMemset(valid, 0, ggraph.vtx_num * sizeof(char)));
    ggraph.valid = valid;
    ggraph.prob_array = prob_array;
    ggraph.alias_array = alias_array;
    H_ERR(cudaMemset(prob_array, 0, ggraph.vtx_num * sizeof(float)));
  }
  void SetSeed(uint _num_seed, uint _hop_num) {
    num_seed = _num_seed;
    std::random_device rd;
    std::mt19937 gen(56);
    std::uniform_int_distribution<> dis(1, 10000); // ggraph.vtx_num);
    uint *seeds = new uint[num_seed];
    for (int n = 0; n < num_seed; ++n) {
      // #ifdef check
      //       // seeds[n] = n;
      //       seeds[n] = 1;
      // #else
      //       seeds[n] = n;
      // // seeds[n] = dis(gen);
      // #endif // check
      seeds[n] = n;
    }
    result.init(num_seed, _hop_num, seeds);
  }
  // void Start();
};

void UnbiasedSample(Sampler sampler);
void UnbiasedWalk(Walker &walker);

void OnlineGBWalk(Walker &walker);
void OnlineGBSample(Sampler sampler);

void StartSP(Sampler sampler);
void Start(Sampler sampler);

void ConstructTable(Sampler &sampler);
// void Sample(Sampler sampler);
void OfflineSample(Sampler &sampler);

// void ConstructTable(Walker &walker);
// void OfflineSample(Walker &walker);
void OfflineWalk(Walker &walker);