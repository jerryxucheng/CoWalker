/*
 * @Description: just perform RW
 * @Date: 2020-11-30 14:30:06
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2021-01-17 21:52:01
 * @FilePath: /skywalker/src/unbiased_sample.cu
 */
#include "app.cuh"

static __global__ void sample_kernel_first(Sampler_new *sampler, uint itr) {
  Jobs_result<JobType::NS, uint> &result = sampler->result;
  gpu_graph *graph = &sampler->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);
  __shared__ matrixBuffer<BLOCK_SIZE, 10, uint> buffer_1hop;
  // __shared__ matrixBuffer<BLOCK_SIZE, 25, uint> buffer_2hop; //not necessary
  // __shared__ uint idxMap[BLOCK_SIZE];
  // idxMap[LTID] = 0;
  buffer_1hop.Init();

  size_t idx_i = TID;
  if (idx_i < result.size)  // for 2-hop, hop_num=3
  {
    // idxMap[LTID] = idx_i;
    uint current_itr = 0;
    coalesced_group active = coalesced_threads();
    // 1-hop
    {
      uint src_id = result.GetData(idx_i, current_itr, 0);
      uint src_degree = graph->getDegree((uint)src_id);
      uint sample_size = MIN(result.hops[current_itr + 1], src_degree);
      for (size_t i = 0; i < sample_size; i++) {
        uint candidate = (int)floor(curand_uniform(&state) * src_degree);
        buffer_1hop.Set(
            graph->getOutNode(src_id, candidate));  // can move back latter
      }
      active.sync();
      buffer_1hop.Flush(result.data + result.length_per_sample * idx_i, 0);
      result.SetSampleLength(idx_i, current_itr, 0, sample_size);
    }
  }
}
template <uint subwarp_size, bool doBuffer = false>
static __global__ void sample_kernel_second(Sampler_new *sampler,
                                            uint current_itr) {}

template <>
static __global__ void sample_kernel_second<16,true>(Sampler_new *sampler,
                                            uint current_itr) {
  Jobs_result<JobType::NS, uint> &result = sampler->result;
  gpu_graph *graph = &sampler->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);
  size_t subwarp_id = TID / 16;
  uint subwarp_idx = TID % 16;
  uint local_subwarp_idx = LTID % 16;
  bool alive = (subwarp_idx < result.hops[current_itr]) ? 1 : 0;
  size_t idx_i = subwarp_id;  //

  __shared__ uint buffer[BLOCK_SIZE][26];
  __shared__ uint buffer_len[BLOCK_SIZE];
  __shared__ uint idxMap[BLOCK_SIZE];
  idxMap[LTID] = 0;
  buffer_len[LTID] = 0;

  if (idx_i < result.size)  // for 2-hop, hop_num=3
  {
    idxMap[LTID] = idx_i;
    coalesced_group active = coalesced_threads();
    {
      uint src_id, src_degree, sample_size;
      if (alive) {
        src_id = result.GetData(idx_i, current_itr, subwarp_idx);
        src_degree = graph->getDegree((uint)src_id);
        sample_size = MIN(result.hops[current_itr + 1], src_degree);
        for (size_t i = 0; i < sample_size; i++) {
          uint candidate = (int)floor(curand_uniform(&state) * src_degree);
          buffer[LTID][i] = graph->getOutNode(src_id, candidate);
          buffer_len[LTID]++;
        }
      }
      if (alive)
        result.SetSampleLength(idx_i, current_itr, subwarp_idx, sample_size);
      active.sync();
      for (size_t i = 0; i < 32; i++) {  // LID id
        for (size_t j = active.thread_rank(); j < buffer_len[(WID)*32 + i];
             j++) {
          *result.GetDataPtr(idxMap[(WID)*32 + i], current_itr + 1,
                             active.thread_rank()) =
              buffer[(WID)*32 + i][active.thread_rank()];
        }
      }
    }
  }
}
template <>
static __global__ void sample_kernel_second<16,false>(Sampler_new *sampler,
                                            uint current_itr) {
  Jobs_result<JobType::NS, uint> &result = sampler->result;
  gpu_graph *graph = &sampler->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);
  size_t subwarp_id = TID / 16;
  uint subwarp_idx = TID % 16;
  uint local_subwarp_idx = LTID % 16;
  bool alive = (subwarp_idx < result.hops[current_itr]) ? 1 : 0;
  size_t idx_i = subwarp_id;  //

  if (idx_i < result.size)  // for 2-hop, hop_num=3
  {
    coalesced_group active = coalesced_threads();
    {
      uint src_id, src_degree, sample_size;
      if (alive) {
        src_id = result.GetData(idx_i, current_itr, subwarp_idx);
        src_degree = graph->getDegree((uint)src_id);
        sample_size = MIN(result.hops[current_itr + 1], src_degree);
        for (size_t i = 0; i < sample_size; i++) {
          uint candidate = (int)floor(curand_uniform(&state) * src_degree);
          *result.GetDataPtr(idx_i, current_itr + 1, i) =
              graph->getOutNode(src_id, candidate);
        }
      }
      if (alive)
        result.SetSampleLength(idx_i, current_itr, subwarp_idx, sample_size);
    }
  }
}

static __global__ void sample_kernel_2hop_buffer(Sampler_new *sampler) {
  Jobs_result<JobType::NS, uint> &result = sampler->result;
  gpu_graph *graph = &sampler->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);
  __shared__ matrixBuffer<BLOCK_SIZE, 10, uint> buffer_1hop;
  // __shared__ matrixBuffer<BLOCK_SIZE, 25, uint> buffer_2hop;  // not
  // necessary
  __shared__ uint idxMap[BLOCK_SIZE];
  idxMap[LTID] = 0;
  buffer_1hop.Init();
  // buffer_2hop.Init();

  size_t idx_i = TID;
  if (idx_i < result.size)  // for 2-hop, hop_num=3
  {
    idxMap[LTID] = idx_i;
    uint current_itr = 0;
    coalesced_group active = coalesced_threads();
    // 1-hop
    {
      uint src_id = result.GetData(idx_i, current_itr, 0);
      uint src_degree = graph->getDegree((uint)src_id);
      uint sample_size = MIN(result.hops[current_itr + 1], src_degree);
      for (size_t i = 0; i < sample_size; i++) {
        uint candidate = (int)floor(curand_uniform(&state) * src_degree);
        buffer_1hop.Set(
            graph->getOutNode(src_id, candidate));  // can move back latter
      }
      active.sync();
      buffer_1hop.Flush(result.data + result.length_per_sample * idx_i, 0);
      result.SetSampleLength(idx_i, current_itr, 0, sample_size);
    }
    current_itr = 1;
    // 2-hop  each warp for one???
    for (size_t i = 0; i < 32; i++) {  // loop over threads
      coalesced_group local = coalesced_threads();
      uint hop1_len;
      if (local.thread_rank() == 0) hop1_len = buffer_1hop.length[(WID)*32 + i];
      hop1_len = local.shfl(hop1_len, 0);

      for (size_t j = 0; j < MIN(result.hops[current_itr], hop1_len);
           j++) {  // loop over 1hop for each thread
        uint src_id =
            buffer_1hop.data[((WID)*32 + i) * buffer_1hop.tileLen + j];
        uint src_degree = graph->getDegree((uint)src_id);
        uint sample_size = MIN(result.hops[current_itr + 1], src_degree);

        for (size_t k = active.thread_rank(); k < sample_size;
             k++) {  // get 2hop for 1hop neighbors for each thread
          uint candidate = (int)floor(curand_uniform(&state) * src_degree);
          *result.GetDataPtr(idxMap[(WID)*32 + i], current_itr + 1,
                             active.thread_rank()) =
              graph->getOutNode(src_id, candidate);
          // buffer_2hop.Set(graph->getOutNode(src_id, candidate));
        }
        // buffer_2hop.Flush(result.data + result.length_per_sample *
        // idxMap[(WID)*32 + i] + j*result.hops[current_itr] , 0);

        if (local.thread_rank() == 0) {
          result.SetSampleLength(idxMap[(WID)*32 + i], current_itr, j,
                                 sample_size);
        }
        local.sync();
      }
    }
  }
}

static __global__ void sample_kernel_2hop(Sampler_new *sampler) {
  Jobs_result<JobType::NS, uint> &result = sampler->result;
  gpu_graph *graph = &sampler->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);
  // if (TID == 0) printf("%s\n", __FUNCTION__);

  size_t idx_i = TID;
  if (idx_i < result.size)  // for 2-hop, hop_num=3
  {
    uint current_itr = 0;
    // 1-hop
    {
      uint src_id = result.GetData(idx_i, current_itr, 0);
      uint src_degree = graph->getDegree((uint)src_id);
      uint sample_size = MIN(result.hops[current_itr + 1], src_degree);
      for (size_t i = 0; i < sample_size; i++) {
        uint candidate = (int)floor(curand_uniform(&state) * src_degree);
        *result.GetDataPtr(idx_i, current_itr + 1, i) =
            graph->getOutNode(src_id, candidate);
        // if (src_id == 1)
        //   printf("add %d\t", graph->getOutNode(src_id, candidate));
      }
      // result.sample_lengths[idx_i*] = sample_size;
      result.SetSampleLength(idx_i, current_itr, 0, sample_size);
    }
    current_itr = 1;
    // 2-hop
    for (size_t k = 0; k < result.hops[current_itr]; k++) {
      uint src_id = result.GetData(idx_i, current_itr, k);
      uint src_degree = graph->getDegree((uint)src_id);
      uint sample_size = MIN(result.hops[current_itr + 1], src_degree);
      for (size_t i = 0; i < sample_size; i++) {
        uint candidate = (int)floor(curand_uniform(&state) * src_degree);
        *result.GetDataPtr(idx_i, current_itr + 1,
                           i + k * result.hops[current_itr]) =
            graph->getOutNode(src_id, candidate);
      }
      // result.sample_lengths[idx_i*result.size_of_sample_lengths+ ] =
      // sample_size;
      result.SetSampleLength(idx_i, current_itr, k, sample_size);
    }
  }
}

static __global__ void print_result(Sampler_new *sampler) {
  sampler->result.PrintResult();
}

float UnbiasedSample(Sampler_new &sampler) {
  LOG("%s\n", __FUNCTION__);
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  Sampler_new *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Sampler_new));
  CUDA_RT_CALL(cudaMemcpy(sampler_ptr, &sampler, sizeof(Sampler_new),
                          cudaMemcpyHostToDevice));
  double start_time, total_time;
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr, false);

  // allocate global buffer
  int block_num = n_sm * FLAGS_m;  // 1024 / BLOCK_SIZE
  CUDA_RT_CALL(cudaDeviceSynchronize());
  CUDA_RT_CALL(cudaPeekAtLastError());

  uint size_h, *size_d;
  cudaMalloc(&size_d, sizeof(uint));
#pragma omp barrier
  start_time = wtime();
  if (FLAGS_peritr) {
    sample_kernel_first<<<sampler.result.size / BLOCK_SIZE + 1, BLOCK_SIZE, 0,
                          0>>>(sampler_ptr, 0);
    sample_kernel_second<16,false>
        <<<sampler.result.size * 16 / BLOCK_SIZE + 1, BLOCK_SIZE, 0, 0>>>(
            sampler_ptr, 1);
  } else {
    if (FLAGS_buffer)
      sample_kernel_2hop_buffer<<<sampler.result.size / BLOCK_SIZE + 1,
                                  BLOCK_SIZE, 0, 0>>>(sampler_ptr);
    else
      sample_kernel_2hop<<<sampler.result.size / BLOCK_SIZE + 1, BLOCK_SIZE, 0,
                           0>>>(sampler_ptr);
  }

  CUDA_RT_CALL(cudaDeviceSynchronize());
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
#pragma omp barrier
  LOG("Device %d sampling time:\t%.6f ratio:\t %.2f MSEPS\n",
      omp_get_thread_num(), total_time,
      static_cast<float>(sampler.result.GetSampledNumber() / total_time /
                         1000000));
  sampler.sampled_edges = sampler.result.GetSampledNumber();
  LOG("sampled_edges %d\n", sampler.sampled_edges);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return total_time;
}
