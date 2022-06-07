/*
 * @Description: just perform RW
 * @Date: 2020-11-30 14:30:06
 * @LastEditors: PengyuWang
 * @LastEditTime: 2021-01-05 23:21:06
 * @FilePath: /sampling/src/offline_walk.cu
 */
#include "app.cuh"
#include </usr/local/cuda-11.6/targets/x86_64-linux/include/nvml.h>
#include "online_walk.cuh"

__global__ void sample_kernel_static_buffer(Walker *walker) {
  Jobs_result<JobType::RW, uint> &result = walker->result;
  gpu_graph *graph = &walker->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);
  __shared__ matrixBuffer<BLOCK_SIZE, 31, uint> buffer;
  buffer.Init();

  size_t idx_i = TID;
  if (idx_i < result.size) {
    result.length[idx_i] = result.hop_num - 1;
    uint src_id;
    // bool alive = true;
    for (uint current_itr = 0; current_itr < result.hop_num - 1;
         current_itr++) {
      if (result.alive[idx_i] != 0) {
        Vector_virtual<uint> alias;
        Vector_virtual<float> prob;
        src_id = current_itr == 0 ? result.GetData(current_itr, idx_i) : src_id;
        uint src_degree = graph->getDegree((uint)src_id);
        alias.Construt(
            graph->alias_array + graph->xadj[src_id] - graph->local_vtx_offset,
            src_degree);
        prob.Construt(
            graph->prob_array + graph->xadj[src_id] - graph->local_vtx_offset,
            src_degree);
        alias.Init(src_degree);
        prob.Init(src_degree);
        const uint target_size = 1;
        if (target_size < src_degree) {
          int col = (int)floor(curand_uniform(&state) * src_degree);
          float p = curand_uniform(&state);
          uint candidate;
          if (p < prob[col])
            candidate = col;
          else
            candidate = alias[col];
          uint next_src = graph->getOutNode(src_id, candidate);
          buffer.Set(next_src);
          src_id = next_src;
        } else if (src_degree == 0) {
          result.alive[idx_i] = 0;
          result.length[idx_i] = current_itr;
          buffer.Finish();
          // return;
        } else {
          uint next_src = graph->getOutNode(src_id, 0);
          buffer.Set(next_src);
          src_id = next_src;
        }
        buffer.CheckFlush(result.data + result.hop_num * idx_i, current_itr);
      }
    }
    buffer.Flush(result.data + result.hop_num * idx_i, 0);
  }
}
// 48 kb , 404 per sampler
__global__ void sample_kernel_static(Walker *walker) {
  Jobs_result<JobType::RW, uint> &result = walker->result;
  gpu_graph *graph = &walker->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);

  size_t idx_i = TID;
  if (idx_i < result.size) {
    result.length[idx_i] = result.hop_num - 1;
    for (uint current_itr = 0; current_itr < result.hop_num - 1;
         current_itr++) {
      if (result.alive[idx_i] != 0) {
        Vector_virtual<uint> alias;
        Vector_virtual<float> prob;
        uint src_id = result.GetData(current_itr, idx_i);
        uint src_degree = graph->getDegree((uint)src_id);
        alias.Construt(
            graph->alias_array + graph->xadj[src_id] - graph->local_vtx_offset,
            src_degree);
        prob.Construt(
            graph->prob_array + graph->xadj[src_id] - graph->local_vtx_offset,
            src_degree);
        alias.Init(src_degree);
        prob.Init(src_degree);
        const uint target_size = 1;
        if (target_size < src_degree) {
          //   int itr = 0;
          // for (size_t i = 0; i < target_size; i++) {
          int col = (int)floor(curand_uniform(&state) * src_degree);
          float p = curand_uniform(&state);
          uint candidate;
          if (p < prob[col])
            candidate = col;
          else
            candidate = alias[col];
          *result.GetDataPtr(current_itr + 1, idx_i) =
              graph->getOutNode(src_id, candidate);
          // }
        } else if (src_degree == 0) {
          result.alive[idx_i] = 0;
          result.length[idx_i] = current_itr;
          break;
        } else {
          *result.GetDataPtr(current_itr + 1, idx_i) =
              graph->getOutNode(src_id, 0);
        }
      }
    }
  }
}

__global__ void sample_kernel(Walker *walker) {
  Jobs_result<JobType::RW, uint> &result = walker->result;
  gpu_graph *graph = &walker->ggraph;
  curandState state;
  curand_init(TID, 0, 0, &state);
  for (size_t idx_i = TID; idx_i < result.size;
       
  

      


       idx_i += gridDim.x * blockDim.x) {
    result.length[idx_i] = result.hop_num - 1;
    for (uint current_itr = 0; current_itr < result.hop_num - 1;
         current_itr++) {
      if (result.alive[idx_i] != 0) {
        Vector_virtual<uint> alias;
        Vector_virtual<float> prob;
        uint src_id = result.GetData(current_itr, idx_i);
        uint src_degree = graph->getDegree((uint)src_id);
        alias.Construt(
            graph->alias_array + graph->xadj[src_id] - graph->local_vtx_offset,
            src_degree);
        prob.Construt(
            graph->prob_array + graph->xadj[src_id] - graph->local_vtx_offset,
            src_degree);
        alias.Init(src_degree);
        prob.Init(src_degree);
        const uint target_size = 1;
        if (target_size < src_degree) {
          //   int itr = 0;
          // for (size_t i = 0; i < target_size; i++) {
          int col = (int)floor(curand_uniform(&state) * src_degree);
          float p = curand_uniform(&state);
          uint candidate;
          if (p < prob[col])
            candidate = col;
          else
            candidate = alias[col];
          *result.GetDataPtr(current_itr + 1, idx_i) =
              graph->getOutNode(src_id, candidate);
          // }
        } else if (src_degree == 0) {
          result.alive[idx_i] = 0;
          result.length[idx_i] = current_itr;
          break;
        } else {
          *result.GetDataPtr(current_itr + 1, idx_i) =
              graph->getOutNode(src_id, 0);
        }
      }
    }
  }
}

static __global__ void print_result(Walker *walker) {
  walker->result.PrintResult();
}

float OfflineWalk(Walker &walker) {
  LOG("%s\n", __FUNCTION__);
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;
  cout<<"num of sms "<< n_sm <<endl;
  Walker *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Walker));
  CUDA_RT_CALL(
      cudaMemcpy(sampler_ptr, &walker, sizeof(Walker), cudaMemcpyHostToDevice));
  double start_time, total_time;
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr);
  // allocate global buffer

  int block_num = FLAGS_sm * FLAGS_m;
  CUDA_RT_CALL(cudaDeviceSynchronize());
  CUDA_RT_CALL(cudaPeekAtLastError());
  start_time = wtime();
#ifdef check
  sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
#else
  if (FLAGS_static) {
    if (FLAGS_buffer)
      // sample_kernel_static_buffer<<<1, 32, 0, 0>>>(sampler_ptr);
      sample_kernel_static_buffer<<<walker.num_seed / BLOCK_SIZE + 1,
                                    BLOCK_SIZE, 0, 0>>>(sampler_ptr);
    else
      sample_kernel_static<<<walker.num_seed / BLOCK_SIZE + 1, BLOCK_SIZE, 0,
                             0>>>(sampler_ptr);
      cout<<"Block number used static: "<<walker.num_seed / BLOCK_SIZE + 1<<endl;
  }
  else{
  //   uint p;
   // sample_kernel<<<FLAGS_sm, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
    // nvmlReturn_t result; 
    // result=nvmlInit();
    // nvmlDevice_t device;
    // uint p1,p2;
    // result=nvmlDeviceGetHandleByIndex(0 , &device); 
    // result=nvmlDeviceGetPowerUsage(device, &p1);
    // // nvmlDeviceSetPowerManagementLimit(device, 1000*p1);
    // cout<<"power: "<< p1<<endl; 
    sample_kernel<<<FLAGS_sm*16, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
    // result=nvmlDeviceGetPowerUsage(device, &p2);
    // // nvmlDeviceSetPowerManagementLimit(device, 1000*p1);
    // cout<<"power: "<< p2<<endl; 
    // cout<<"Block number used not static: "<<block_num<<endl;
  }
#endif

  CUDA_RT_CALL(cudaDeviceSynchronize());
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  LOG("Device %d sampling time:\t%.6f ratio:\t %.2f MSEPS\n",
      omp_get_thread_num(), total_time,
      static_cast<float>(walker.result.GetSampledNumber() / total_time /
                         1000000));
  walker.sampled_edges = walker.result.GetSampledNumber();
  LOG("sampled_edges %d\n", walker.sampled_edges);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return total_time;
}


float OfflineWalk2(Walker &walker, Walker &walker2) {
  LOG("%s\n", __FUNCTION__);
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;
  cout<<"num of sms "<< n_sm <<endl;
  Walker *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Walker));
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr, &walker, sizeof(Walker), cudaMemcpyHostToDevice,stream1));
  double start_time, total_time;
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr);
  // allocate global buffer

  Walker *sampler_ptr2;
  cudaMalloc(&sampler_ptr2, sizeof(Walker));
  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr2, &walker2, sizeof(Walker), cudaMemcpyHostToDevice,stream2));
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr2);
  // allocate global buffer


  double t2,t1;
  int block_num = FLAGS_sm * FLAGS_m;
  CUDA_RT_CALL(cudaDeviceSynchronize());
  CUDA_RT_CALL(cudaPeekAtLastError());
  start_time = wtime();
#ifdef check
  sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
#else
  if (FLAGS_static) {
    if (FLAGS_buffer)
      // sample_kernel_static_buffer<<<1, 32, 0, 0>>>(sampler_ptr);
      sample_kernel_static_buffer<<<walker.num_seed / BLOCK_SIZE + 1,
                                    BLOCK_SIZE, 0, 0>>>(sampler_ptr);
    else
      sample_kernel_static<<<walker.num_seed / BLOCK_SIZE + 1, BLOCK_SIZE, 0,
                             0>>>(sampler_ptr);
      cout<<"Block number used static: "<<walker.num_seed / BLOCK_SIZE + 1<<endl;
  }
  else{  
    if (FLAGS_k1){
    sample_kernel<<<FLAGS_sm*4, BLOCK_SIZE, 0, stream1>>>(sampler_ptr);
    }
    else if (FLAGS_k2){
    sample_kernel<<<FLAGS_sm*4, BLOCK_SIZE, 0, stream2>>>(sampler_ptr2);

    }
    else{
    sample_kernel<<<FLAGS_sm/2, BLOCK_SIZE, 0, stream1>>>(sampler_ptr);
    sample_kernel<<<FLAGS_sm/2, BLOCK_SIZE, 0, stream2>>>(sampler_ptr2);
    }
    }
#endif
  
  CUDA_RT_CALL(cudaDeviceSynchronize());  
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  cout<<"time used running two kernels after synchronize :"<<total_time<<endl;
  LOG("Device %d sampling time:\t%.6f ratio:\t %.2f MSEPS\n",
      omp_get_thread_num(), total_time,
      static_cast<float>(walker.result.GetSampledNumber() / total_time /
                         1000000));
  walker.sampled_edges = walker.result.GetSampledNumber();
  LOG("sampled_edges %d\n", walker.sampled_edges);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  return total_time;
}


float OfflineWalk3(Walker &walker, Walker &walker2, Walker &walker3) {
  LOG("%s\n", __FUNCTION__);
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;
  cout<<"num of sms "<< n_sm <<endl;
  Walker *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Walker));
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr, &walker, sizeof(Walker), cudaMemcpyHostToDevice,stream1));
  double start_time, total_time;
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr);
  // allocate global buffer

  Walker *sampler_ptr2;
  cudaMalloc(&sampler_ptr2, sizeof(Walker));
  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr2, &walker2, sizeof(Walker), cudaMemcpyHostToDevice,stream2));
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr2);
  // allocate global buffer
  
  Walker *sampler_ptr3;
  cudaMalloc(&sampler_ptr3, sizeof(Walker));
  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr3, &walker3, sizeof(Walker), cudaMemcpyHostToDevice,stream2));
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr3);
  // allocate global buffer


  double t2,t1;
  int block_num = FLAGS_sm * FLAGS_m;
  CUDA_RT_CALL(cudaDeviceSynchronize());
  CUDA_RT_CALL(cudaPeekAtLastError());
  start_time = wtime();
#ifdef check
  sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
#else
  if (FLAGS_static) {
    if (FLAGS_buffer)
      // sample_kernel_static_buffer<<<1, 32, 0, 0>>>(sampler_ptr);
      sample_kernel_static_buffer<<<walker.num_seed / BLOCK_SIZE + 1,
                                    BLOCK_SIZE, 0, 0>>>(sampler_ptr);
    else
      sample_kernel_static<<<walker.num_seed / BLOCK_SIZE + 1, BLOCK_SIZE, 0,
                             0>>>(sampler_ptr);
      cout<<"Block number used static: "<<walker.num_seed / BLOCK_SIZE + 1<<endl;
  }
  else{  
    if (FLAGS_k1){
    sample_kernel<<<FLAGS_sm, BLOCK_SIZE, 0, stream1>>>(sampler_ptr);
    }
    else if (FLAGS_k2){
    sample_kernel<<<FLAGS_sm, BLOCK_SIZE, 0, stream2>>>(sampler_ptr2);

    }
    else{
    sample_kernel<<<FLAGS_sm/4, BLOCK_SIZE, 0, stream1>>>(sampler_ptr);
    sample_kernel<<<FLAGS_sm*3/4, BLOCK_SIZE, 0, stream2>>>(sampler_ptr2);
    sample_kernel<<<FLAGS_sm*3/4, BLOCK_SIZE, 0, stream2>>>(sampler_ptr3);
    }
    }
#endif
  
  CUDA_RT_CALL(cudaDeviceSynchronize());  
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  cout<<"time used running two kernels after synchronize :"<<total_time<<endl;
  LOG("Device %d sampling time:\t%.6f ratio:\t %.2f MSEPS\n",
      omp_get_thread_num(), total_time,
      static_cast<float>(walker.result.GetSampledNumber() / total_time /
                         1000000));
  walker.sampled_edges = walker.result.GetSampledNumber();
  LOG("sampled_edges %d\n", walker.sampled_edges);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  return total_time;
}


float OfflineWalk4(Walker &walker, Walker &walker2, Walker &walker3, Walker &walker4) {
  LOG("%s\n", __FUNCTION__);
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;
  cout<<"num of sms "<< n_sm <<endl;
  Walker *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Walker));
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr, &walker, sizeof(Walker), cudaMemcpyHostToDevice,stream1));
  double start_time, total_time;
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr);
  // allocate global buffer

  Walker *sampler_ptr2;
  cudaMalloc(&sampler_ptr2, sizeof(Walker));
  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr2, &walker2, sizeof(Walker), cudaMemcpyHostToDevice,stream2));
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr2);
  // allocate global buffer
  
  Walker *sampler_ptr3;
  cudaMalloc(&sampler_ptr3, sizeof(Walker));
  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr3, &walker3, sizeof(Walker), cudaMemcpyHostToDevice,stream2));
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr3);
  // allocate global buffer
  
   Walker *sampler_ptr4;
  cudaMalloc(&sampler_ptr4, sizeof(Walker));
  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr4, &walker4, sizeof(Walker), cudaMemcpyHostToDevice,stream2));
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr4);
  // allocate global buffer


  double t2,t1;
  int block_num = FLAGS_sm * FLAGS_m;
  CUDA_RT_CALL(cudaDeviceSynchronize());
  CUDA_RT_CALL(cudaPeekAtLastError());
  start_time = wtime();
#ifdef check
  sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
#else
  if (FLAGS_static) {
    if (FLAGS_buffer)
      // sample_kernel_static_buffer<<<1, 32, 0, 0>>>(sampler_ptr);
      sample_kernel_static_buffer<<<walker.num_seed / BLOCK_SIZE + 1,
                                    BLOCK_SIZE, 0, 0>>>(sampler_ptr);
    else
      sample_kernel_static<<<walker.num_seed / BLOCK_SIZE + 1, BLOCK_SIZE, 0,
                             0>>>(sampler_ptr);
      cout<<"Block number used static: "<<walker.num_seed / BLOCK_SIZE + 1<<endl;
  }
  else{  
    if (FLAGS_k1){
    sample_kernel<<<FLAGS_sm, BLOCK_SIZE, 0, stream1>>>(sampler_ptr);
    }
    else if (FLAGS_k2){
    sample_kernel<<<FLAGS_sm, BLOCK_SIZE, 0, stream2>>>(sampler_ptr2);

    }
    else{
    sample_kernel<<<FLAGS_sm/4, BLOCK_SIZE, 0, stream1>>>(sampler_ptr);
    sample_kernel<<<FLAGS_sm*3/4, BLOCK_SIZE, 0, stream2>>>(sampler_ptr2);
    sample_kernel<<<FLAGS_sm*3/4, BLOCK_SIZE, 0, stream2>>>(sampler_ptr3);
    sample_kernel<<<FLAGS_sm*3/4, BLOCK_SIZE, 0, stream2>>>(sampler_ptr4);
    // print_result<<<1, 32, 0, 0>>>(sampler_ptr);
    // print_result<<<1, 32, 0, 0>>>(sampler_ptr2);
    // print_result<<<1, 32, 0, 0>>>(sampler_ptr3);
    // print_result<<<1, 32, 0, 0>>>(sampler_ptr4);
    }
    }
#endif
  
  CUDA_RT_CALL(cudaDeviceSynchronize());  
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  cout<<"time used running two kernels after synchronize :"<<total_time<<endl;
  LOG("Device %d sampling time:\t%.6f ratio:\t %.2f MSEPS\n",
      omp_get_thread_num(), total_time,
      static_cast<float>(walker.result.GetSampledNumber() / total_time /
                         1000000));
  walker.sampled_edges = walker.result.GetSampledNumber();
  LOG("sampled_edges %d\n", walker.sampled_edges);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  return total_time;
}


float OfflineWalk5(Walker &walker, Walker &walker2, Walker &walker3, Walker &walker4, Walker &walker5) {
  LOG("%s\n", __FUNCTION__);
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;
  cout<<"num of sms "<< n_sm <<endl;
  Walker *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Walker));
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr, &walker, sizeof(Walker), cudaMemcpyHostToDevice,stream1));
  double start_time, total_time;
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr);
  // allocate global buffer

  Walker *sampler_ptr2;
  cudaMalloc(&sampler_ptr2, sizeof(Walker));
  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr2, &walker2, sizeof(Walker), cudaMemcpyHostToDevice,stream2));
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr2);
  // allocate global buffer
  
  Walker *sampler_ptr3;
  cudaMalloc(&sampler_ptr3, sizeof(Walker));
  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr3, &walker3, sizeof(Walker), cudaMemcpyHostToDevice,stream2));
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr3);
  // allocate global buffer
  
   Walker *sampler_ptr4;
  cudaMalloc(&sampler_ptr4, sizeof(Walker));
  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr4, &walker4, sizeof(Walker), cudaMemcpyHostToDevice,stream2));
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr4);
  // allocate global buffer

  Walker *sampler_ptr5;
  cudaMalloc(&sampler_ptr5, sizeof(Walker));
  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr5, &walker4, sizeof(Walker), cudaMemcpyHostToDevice,stream2));
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr5);
  // allocate global buffer


  double t2,t1;
  int block_num = FLAGS_sm * FLAGS_m;
  CUDA_RT_CALL(cudaDeviceSynchronize());
  CUDA_RT_CALL(cudaPeekAtLastError());
  start_time = wtime();
#ifdef check
  sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
#else
  if (FLAGS_static) {
    if (FLAGS_buffer)
      // sample_kernel_static_buffer<<<1, 32, 0, 0>>>(sampler_ptr);
      sample_kernel_static_buffer<<<walker.num_seed / BLOCK_SIZE + 1,
                                    BLOCK_SIZE, 0, 0>>>(sampler_ptr);
    else
      sample_kernel_static<<<walker.num_seed / BLOCK_SIZE + 1, BLOCK_SIZE, 0,
                             0>>>(sampler_ptr);
      cout<<"Block number used static: "<<walker.num_seed / BLOCK_SIZE + 1<<endl;
  }
  else{  
    if (FLAGS_k1){
    sample_kernel<<<FLAGS_sm/2, BLOCK_SIZE, 0, stream1>>>(sampler_ptr);
    }
    else if (FLAGS_k2){
    sample_kernel<<<FLAGS_sm/2, BLOCK_SIZE, 0, stream2>>>(sampler_ptr2);

    }
    else{
    sample_kernel<<<FLAGS_sm/4, BLOCK_SIZE, 0, stream1>>>(sampler_ptr);
    sample_kernel<<<FLAGS_sm*3/4, BLOCK_SIZE, 0, stream2>>>(sampler_ptr2);
    sample_kernel<<<FLAGS_sm*3/4, BLOCK_SIZE, 0, stream2>>>(sampler_ptr3);
    sample_kernel<<<FLAGS_sm*3/4, BLOCK_SIZE, 0, stream2>>>(sampler_ptr4);
    sample_kernel<<<FLAGS_sm*3/4, BLOCK_SIZE, 0, stream2>>>(sampler_ptr5);
    }
    }
#endif
  
  CUDA_RT_CALL(cudaDeviceSynchronize());  
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  cout<<"time used running two kernels after synchronize :"<<total_time<<endl;
  LOG("Device %d sampling time:\t%.6f ratio:\t %.2f MSEPS\n",
      omp_get_thread_num(), total_time,
      static_cast<float>(walker.result.GetSampledNumber() / total_time /
                         1000000));
  walker.sampled_edges = walker.result.GetSampledNumber();
  LOG("sampled_edges %d\n", walker.sampled_edges);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  return total_time;
}


// float MixWalk(Walker &walker, Walker &sampler) {
//   LOG("%s\n", __FUNCTION__);
//   int device;
//   cudaDeviceProp prop;
//   cudaGetDevice(&device);
//   cudaGetDeviceProperties(&prop, device);
//   int n_sm = prop.multiProcessorCount;
//   cout<<"num of sms "<< n_sm <<endl;
//   Walker *sampler_ptr;
//   cudaMalloc(&sampler_ptr, sizeof(Walker));
//   cudaStream_t stream1, stream2;
//   cudaStreamCreate(&stream1);
//   cudaStreamCreate(&stream2);

//   CUDA_RT_CALL(
//       cudaMemcpyAsync(sampler_ptr, &walker, sizeof(Walker), cudaMemcpyHostToDevice,stream1));
//   double start_time, total_time;
//   // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
//   BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr);
//   // allocate global buffer

  
  
//   Walker *sampler_ptr2;
//   cudaMalloc(&sampler_ptr2, sizeof(Walker));
//   CUDA_RT_CALL(cudaMemcpy(sampler_ptr2, &sampler, sizeof(Walker),
//                           cudaMemcpyHostToDevice));
//   init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr2, true);
//   BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr2);
//   init_array(sampler.result.length, sampler.result.size,
//              sampler.result.hop_num);
//   // allocate global buffer
//   int block_num = FLAGS_sm;
//   int gbuff_size = sampler.ggraph.MaxDegree;
//   LOG("alllocate GMEM buffer %d MB\n",
//       block_num * gbuff_size * MEM_PER_ELE / 1024 / 1024);

//   Vector_pack<uint> *vector_pack_h = new Vector_pack<uint>[block_num];
//   for (size_t i = 0; i < block_num; i++) {
//     vector_pack_h[i].Allocate(gbuff_size, sampler.device_id);
//   }
//   CUDA_RT_CALL(cudaDeviceSynchronize());
//   Vector_pack<uint> *vector_packs;
//   CUDA_RT_CALL(
//       cudaMalloc(&vector_packs, sizeof(Vector_pack<uint>) * block_num));
//   CUDA_RT_CALL(cudaMemcpy(vector_packs, vector_pack_h,
//                           sizeof(Vector_pack<uint>) * block_num,
//                           cudaMemcpyHostToDevice));


//   double t2,t1;
//   int block_num = FLAGS_sm * FLAGS_m;
//   CUDA_RT_CALL(cudaDeviceSynchronize());
//   CUDA_RT_CALL(cudaPeekAtLastError());
//   start_time = wtime();
// #ifdef check
//   sample_kernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr);
// #else
//   if (FLAGS_static) {
//     if (FLAGS_buffer)
//       // sample_kernel_static_buffer<<<1, 32, 0, 0>>>(sampler_ptr);
//       sample_kernel_static_buffer<<<walker.num_seed / BLOCK_SIZE + 1,
//                                     BLOCK_SIZE, 0, 0>>>(sampler_ptr);
//     else
//       sample_kernel_static<<<walker.num_seed / BLOCK_SIZE + 1, BLOCK_SIZE, 0,
//                              0>>>(sampler_ptr);
//       cout<<"Block number used static: "<<walker.num_seed / BLOCK_SIZE + 1<<endl;
//   }
//   else{  
//     if (FLAGS_k1){
//     sample_kernel<<<FLAGS_sm/2, BLOCK_SIZE, 0, stream1>>>(sampler_ptr);
//     }
//     // else if (FLAGS_k2){
//     // sample_kernel<<<FLAGS_sm/2, BLOCK_SIZE, 0, stream2>>>(sampler_ptr2);

//     // }
//     else{
//     sample_kernel<<<FLAGS_sm/4, BLOCK_SIZE, 0, stream1>>>(sampler_ptr);
//     OnlineWalkKernel<<<FLAGS_sm*4, BLOCK_SIZE, 0, stream2>>>(sampler_ptr2,
//                                                         vector_packs, 0);
//     }
//     }
// #endif
  
//   CUDA_RT_CALL(cudaDeviceSynchronize());  
//   // CUDA_RT_CALL(cudaPeekAtLastError());
//   total_time = wtime() - start_time;
//   cout<<"time used running two kernels after synchronize :"<<total_time<<endl;
//   LOG("Device %d sampling time:\t%.6f ratio:\t %.2f MSEPS\n",
//       omp_get_thread_num(), total_time,
//       static_cast<float>(walker.result.GetSampledNumber() / total_time /
//                          1000000));
//   walker.sampled_edges = walker.result.GetSampledNumber();
//   LOG("sampled_edges %d\n", walker.sampled_edges);
//   if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
//   CUDA_RT_CALL(cudaDeviceSynchronize());
//   cudaStreamDestroy(stream1);
//   cudaStreamDestroy(stream2);
//   return total_time;
// }







// Online walk starts here 


/*
 * @Description: online walk.
 * @Date: 2020-12-06 17:29:39
 * @LastEditors: PengyuWang
 * @LastEditTime: 2021-01-11 16:56:20
 * @FilePath: /skywalker/src/online_walk.cu
 */


static __device__ void SampleWarpCentic(Jobs_result<JobType::RW, uint> &result,
                                        gpu_graph *ggraph, curandState state,
                                        int current_itr, int node_id,
                                        void *buffer, uint instance_id = 0) {
  alias_table_constructor_shmem<uint, thread_block_tile<32>> *tables =
      (alias_table_constructor_shmem<uint, thread_block_tile<32>> *)buffer;
  alias_table_constructor_shmem<uint, thread_block_tile<32>> *table =
      &tables[WID];
  bool not_all_zero = table->loadFromGraph(ggraph->getNeighborPtr(node_id),
                                           ggraph, ggraph->getDegree(node_id),
                                           current_itr, node_id, instance_id);
  if (not_all_zero) {
    table->construct();
    if (LID == 0) {
      int col = (int)floor(curand_uniform(&state) * table->Size());
      float p = curand_uniform(&state);
      uint candidate;
      if (p < table->GetProb(col))
        candidate = col;
      else
        candidate = table->GetAlias(col);
      result.AddActive(current_itr, instance_id);
      *result.GetDataPtr(current_itr + 1, instance_id) =
          ggraph->getOutNode(node_id, candidate);
      ggraph->UpdateWalkerState(instance_id, node_id);
    }
  } else {
    if (LID == 0) result.length[instance_id] = current_itr;
  }
  table->Clean();
}

static __device__ void SampleBlockCentic(Jobs_result<JobType::RW, uint> &result,
                                         gpu_graph *ggraph, curandState state,
                                         int current_itr, int node_id,
                                         void *buffer,
                                         Vector_pack<uint> *vector_packs,
                                         uint instance_id = 0) {
  alias_table_constructor_shmem<uint, thread_block, BufferType::GMEM> *tables =
      (alias_table_constructor_shmem<uint, thread_block, BufferType::GMEM> *)
          buffer;
  alias_table_constructor_shmem<uint, thread_block, BufferType::GMEM> *table =
      &tables[0];
  table->loadGlobalBuffer(vector_packs);
  __syncthreads();
  bool not_all_zero = table->loadFromGraph(ggraph->getNeighborPtr(node_id),
                                           ggraph, ggraph->getDegree(node_id),
                                           current_itr, node_id, instance_id);
  __syncthreads();
  if (not_all_zero) {
    table->constructBC();
    __syncthreads();
    if (LTID == 0) {
      int col = (int)floor(curand_uniform(&state) * table->Size());
      float p = curand_uniform(&state);
      uint candidate;
      if (p < table->GetProb(col))
        candidate = col;
      else
        candidate = table->GetAlias(col);
      result.AddActive(current_itr, instance_id);
      *result.GetDataPtr(current_itr + 1, instance_id) =
          ggraph->getOutNode(node_id, candidate);
      ggraph->UpdateWalkerState(instance_id, node_id);
    };
  } else {
    if (LTID == 0) result.length[instance_id] = current_itr;
  }
  __syncthreads();
  table->Clean();
}

static __global__ void OnlineWalkKernel(Walker *sampler,
                                 Vector_pack<uint> *vector_pack, float *tp) {
  Jobs_result<JobType::RW, uint> &result = sampler->result;
  gpu_graph *ggraph = &sampler->ggraph;
  Vector_pack<uint> *vector_packs = &vector_pack[BID];
  __shared__ alias_table_constructor_shmem<uint, thread_block_tile<32>>
      table[WARP_PER_BLK];
  void *buffer = &table[0];
  curandState state;
  curand_init(TID, 0, 0, &state);

  __shared__ uint current_itr;
  if (threadIdx.x == 0) current_itr = 0;
  __syncthreads();
  for (; current_itr < result.hop_num - 1;) {
    sample_job_new job;
    __threadfence_block();
    if (LID == 0) {
      job = result.requireOneJob(current_itr);
    }
    __syncwarp(FULL_WARP_MASK);
    job.val = __shfl_sync(FULL_WARP_MASK, job.val, 0);
    job.instance_idx = __shfl_sync(FULL_WARP_MASK, job.instance_idx, 0);
    __syncwarp(FULL_WARP_MASK);
    while (job.val) {
      uint node_id = result.GetData(current_itr, job.instance_idx);
      bool stop =
          __shfl_sync(FULL_WARP_MASK, (curand_uniform(&state) < *tp), 0);
      if (!stop) {
        if (ggraph->getDegree(node_id) < ELE_PER_WARP) {
          SampleWarpCentic(result, ggraph, state, current_itr, node_id, buffer,
                           job.instance_idx);
        } else {
#ifdef skip8k
          if (LID == 0 && ggraph->getDegree(node_id) < 8000)
#else
          if (LID == 0)
#endif  // skip8k
            result.AddHighDegree(current_itr, job.instance_idx);
        }
      } else {
        if (LID == 0) result.length[job.instance_idx] = current_itr;
      }
      __syncwarp(FULL_WARP_MASK);
      if (LID == 0) job = result.requireOneJob(current_itr);
      __syncwarp(FULL_WARP_MASK);
      job.val = __shfl_sync(FULL_WARP_MASK, job.val, 0);
      job.instance_idx = __shfl_sync(FULL_WARP_MASK, job.instance_idx, 0);
      __syncwarp(FULL_WARP_MASK);
    }
    __syncthreads();
    __shared__ sample_job_new high_degree_job;  // really use job_id
    __shared__ uint node_id;
    if (LTID == 0) {
      sample_job_new tmp = result.requireOneHighDegreeJob(current_itr);
      high_degree_job.val = tmp.val;
      high_degree_job.instance_idx = tmp.instance_idx;
      if (tmp.val) {
        node_id = result.GetData(current_itr, high_degree_job.instance_idx);
      }
    }
    __syncthreads();
    while (high_degree_job.val) {
      SampleBlockCentic(result, ggraph, state, current_itr, node_id, buffer,
                        vector_packs,
                        high_degree_job.instance_idx);  // buffer_pointer
      __syncthreads();
      if (LTID == 0) {
        sample_job_new tmp = result.requireOneHighDegreeJob(current_itr);
        high_degree_job.val = tmp.val;
        high_degree_job.instance_idx = tmp.instance_idx;
        if (high_degree_job.val) {
          node_id = result.GetData(current_itr, high_degree_job.instance_idx);
        }
      }
      __syncthreads();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      result.NextItr(current_itr);
    }
    __syncthreads();
  }
}
__device__ uint get_smid() {
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret));
  return ret;
}
static __global__ void OnlineWalkKernelStatic(Walker *sampler,
                                       Vector_pack<uint> *vector_pack,
                                       uint current_itr, float *tp,
                                       uint n = 1) {
  Jobs_result<JobType::RW, uint> &result = sampler->result;
  gpu_graph *ggraph = &sampler->ggraph;
  Vector_pack<uint> *vector_packs = &vector_pack[blockIdx.x];
  __shared__ alias_table_constructor_shmem<uint, thread_block_tile<32>>
      table[WARP_PER_BLK];
  void *buffer = &table[0];
  __shared__ Vector_shmem<uint, thread_block, BLOCK_SIZE, false>
      local_high_degree;
  local_high_degree.Init();

  curandState state;
  curand_init(TID, 0, 0, &state);
  // size_t idx = GWID;
  // if (idx < result.job_sizes[current_itr])
  for (size_t idx = GWID; idx < result.size;
       idx += blockDim.x / 32 * gridDim.x) {
    if (result.length[idx] == result.hop_num) {  // walker is alive
      // if (LID == 0) printf("instance_id %d\n", idx);
      size_t node_id = result.GetData(current_itr, idx);
      uint src_degree = ggraph->getDegree(node_id);
      bool stop =
          __shfl_sync(FULL_WARP_MASK, (curand_uniform(&state) < *tp), 0);
      if (!stop) {
        if (src_degree == 1) {
          if (LID == 0) {
            *result.GetDataPtr(current_itr + 1, idx) =
                ggraph->getOutNode(node_id, 0);
            ggraph->UpdateWalkerState(idx, node_id);
          }
        } else if (src_degree == 0) {
          if (LID == 0) result.length[idx] = current_itr;
        } else if (src_degree < ELE_PER_WARP) {
          SampleWarpCentic(result, ggraph, state, current_itr, node_id, buffer,
                           idx);
          //  if (LID == 0) result.length[idx]= current_itr+1;
        } else {
          if (LID == 0) local_high_degree.Add(idx);
        }
      } else {
        if (LID == 0) result.length[idx] = current_itr;
      }
    }
  }
  __syncthreads();
  for (size_t i = 0; i < local_high_degree.Size(); i++) {
    __syncthreads();
    size_t node_id = result.GetData(current_itr, local_high_degree.Get(i));
    SampleBlockCentic(result, ggraph, state, current_itr, node_id, buffer,
                      vector_packs, local_high_degree.Get(i));
    // if (LTID == 0) result.length[local_high_degree.Get(i)]= current_itr+1;
  }
}


// static __global__ void print_result(Walker *sampler) {
//   sampler->result.PrintResult();
// }

template <typename T>
__global__ void init_array_d(T *ptr, size_t size, T v) {
  if (TID < size) {
    ptr[TID] = v;
  }
}
template <typename T>
void init_array(T *ptr, size_t size, T v) {
  init_array_d<T><<<size / 512 + 1, 512>>>(ptr, size, v);
}

// void Start_high_degree(Walker sampler)
float OnlineWalkShMem(Walker &sampler) {
  // orkut max degree 932101
  LOG("%s\n", __FUNCTION__);
#ifdef skip8k
  LOG("skipping 8k\n");
#endif  // skip8k

  LOG("overring staic flag, static\n");
  FLAGS_static=0;

  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  Walker *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Walker));
  CUDA_RT_CALL(cudaMemcpy(sampler_ptr, &sampler, sizeof(Walker),
                          cudaMemcpyHostToDevice));
  double start_time, total_time;
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr, true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr);
  init_array(sampler.result.length, sampler.result.size,
             sampler.result.hop_num);
  // allocate global buffer
  int block_num = FLAGS_sm;
  int gbuff_size = sampler.ggraph.MaxDegree;
  ;
  LOG("alllocate GMEM buffer %d MB\n",
      block_num * gbuff_size * MEM_PER_ELE / 1024 / 1024);

  Vector_pack<uint> *vector_pack_h = new Vector_pack<uint>[block_num];
  for (size_t i = 0; i < block_num; i++) {
    vector_pack_h[i].Allocate(gbuff_size, sampler.device_id);
  }
  CUDA_RT_CALL(cudaDeviceSynchronize());
  Vector_pack<uint> *vector_packs;
  CUDA_RT_CALL(
      cudaMalloc(&vector_packs, sizeof(Vector_pack<uint>) * block_num));
  CUDA_RT_CALL(cudaMemcpy(vector_packs, vector_pack_h,
                          sizeof(Vector_pack<uint>) * block_num,
                          cudaMemcpyHostToDevice));

  float *tp_d, tp;
  tp = FLAGS_tp;
  cudaMalloc(&tp_d, sizeof(float));
  CUDA_RT_CALL(cudaMemcpy(tp_d, &tp, sizeof(float), cudaMemcpyHostToDevice));

  //  Global_buffer
  CUDA_RT_CALL(cudaDeviceSynchronize());
  start_time = wtime();
  if (FLAGS_static) {
    for (size_t i = 0; i < sampler.result.hop_num - 1; i++) {
      OnlineWalkKernelStatic<<<block_num, BLOCK_SIZE, 0, 0>>>(
          sampler_ptr, vector_packs, i, tp_d, FLAGS_m);
      CUDA_RT_CALL(cudaDeviceSynchronize());
      CUDA_RT_CALL(cudaPeekAtLastError());
    }

  } else {
    if (FLAGS_debug)
      OnlineWalkKernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs,
                                                tp_d);
    else{
      OnlineWalkKernel<<<block_num, BLOCK_SIZE, 0, 0>>>(sampler_ptr,
                                                        vector_packs, tp_d);
      cout<<"blocks used online walk kernel: "<<block_num<<endl;
      cout<<"online"<<endl;
      print_result<<<1, 32, 0, 0>>>(sampler_ptr);
    }
     
  }

  CUDA_RT_CALL(cudaDeviceSynchronize());
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  LOG("Device %d sampling time:\t%.2f ms ratio:\t %.1f MSEPS\n",
      omp_get_thread_num(), total_time * 1000,
      static_cast<float>(sampler.result.GetSampledNumber() / total_time /
                         1000000));
  sampler.sampled_edges = sampler.result.GetSampledNumber();
  LOG("sampled_edges %d\n", sampler.sampled_edges);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return total_time;
}






float OnlineWalkShMem2(Walker &sampler, Walker &sampler2) {
  // orkut max degree 932101
  LOG("%s\n", __FUNCTION__);
#ifdef skip8k
  LOG("skipping 8k\n");
#endif  // skip8k

  LOG("overring staic flag, static\n");
  FLAGS_static=0;
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  Walker *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Walker));
  CUDA_RT_CALL(cudaMemcpy(sampler_ptr, &sampler, sizeof(Walker),
                          cudaMemcpyHostToDevice));
  double start_time, total_time;
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr, true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr);
  init_array(sampler.result.length, sampler.result.size,
             sampler.result.hop_num);
  // allocate global buffer
  int block_num = FLAGS_sm;
  int gbuff_size = sampler.ggraph.MaxDegree;

  LOG("alllocate GMEM buffer %d MB\n",
      block_num * gbuff_size * MEM_PER_ELE / 1024 / 1024);

  Vector_pack<uint> *vector_pack_h = new Vector_pack<uint>[block_num];
  for (size_t i = 0; i < block_num; i++) {
    vector_pack_h[i].Allocate(gbuff_size, sampler.device_id);
  }
  CUDA_RT_CALL(cudaDeviceSynchronize());
  Vector_pack<uint> *vector_packs;
  CUDA_RT_CALL(
      cudaMalloc(&vector_packs, sizeof(Vector_pack<uint>) * block_num));
  CUDA_RT_CALL(cudaMemcpy(vector_packs, vector_pack_h,
                          sizeof(Vector_pack<uint>) * block_num,
                          cudaMemcpyHostToDevice));

  float *tp_d, tp;
  tp = FLAGS_tp;
  cudaMalloc(&tp_d, sizeof(float));
  CUDA_RT_CALL(cudaMemcpy(tp_d, &tp, sizeof(float), cudaMemcpyHostToDevice));
 



  Walker *sampler_ptr2;
  cudaMalloc(&sampler_ptr2, sizeof(Walker));
  CUDA_RT_CALL(cudaMemcpy(sampler_ptr2, &sampler2, sizeof(Walker),
                          cudaMemcpyHostToDevice));
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr2, true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr2);
  init_array(sampler2.result.length, sampler2.result.size,
             sampler2.result.hop_num);
  // allocate global buffer
  block_num = FLAGS_sm;
  gbuff_size = sampler2.ggraph.MaxDegree;
  
  LOG("alllocate GMEM buffer %d MB\n",
      block_num * gbuff_size * MEM_PER_ELE / 1024 / 1024);

  Vector_pack<uint> *vector_pack_h2 = new Vector_pack<uint>[block_num];
  for (size_t i = 0; i < block_num; i++) {
    vector_pack_h2[i].Allocate(gbuff_size, sampler.device_id);
  }
  CUDA_RT_CALL(cudaDeviceSynchronize());
  Vector_pack<uint> *vector_packs2;
  CUDA_RT_CALL(
      cudaMalloc(&vector_packs2, sizeof(Vector_pack<uint>) * block_num));
  CUDA_RT_CALL(cudaMemcpy(vector_packs2, vector_pack_h2,
                          sizeof(Vector_pack<uint>) * block_num,
                          cudaMemcpyHostToDevice));

 

  //  Global_buffer
  CUDA_RT_CALL(cudaDeviceSynchronize());
  start_time = wtime();
  if (FLAGS_static) {
    for (size_t i = 0; i < sampler2.result.hop_num - 1; i++) {
      OnlineWalkKernelStatic<<<block_num, BLOCK_SIZE, 0, 0>>>(
          sampler_ptr2, vector_packs2, i, tp_d, FLAGS_m);
      CUDA_RT_CALL(cudaDeviceSynchronize());
      CUDA_RT_CALL(cudaPeekAtLastError());
    }

  } else {
    if (FLAGS_debug)
      OnlineWalkKernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs,
                                                tp_d);
    else
      OnlineWalkKernel<<<block_num/2, BLOCK_SIZE, 0, stream1>>>(sampler_ptr,
                                                        vector_packs, tp_d);
      OnlineWalkKernel<<<block_num/2, BLOCK_SIZE, 0, stream2>>>(sampler_ptr2,
                                                        vector_packs2, tp_d);
      cout<<"blocks: "<<block_num<<endl;
  }

  CUDA_RT_CALL(cudaDeviceSynchronize());
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  LOG("Device %d sampling time:\t%.2f ms ratio:\t %.1f MSEPS\n",
      omp_get_thread_num(), total_time * 1000,
      static_cast<float>(sampler.result.GetSampledNumber() / total_time /
                         1000000));
  sampler2.sampled_edges = sampler.result.GetSampledNumber();
  LOG("sampled_edges %d\n", sampler.sampled_edges);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return total_time;
}


float OnlineWalkShMem4(Walker &sampler, Walker &sampler2, Walker &sampler3, Walker &sampler4) {
  // orkut max degree 932101
  LOG("%s\n", __FUNCTION__);
#ifdef skip8k
  LOG("skipping 8k\n");
#endif  // skip8k

  LOG("overring staic flag, static\n");
  FLAGS_static=0;
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;

  Walker *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Walker));
  CUDA_RT_CALL(cudaMemcpy(sampler_ptr, &sampler, sizeof(Walker),
                          cudaMemcpyHostToDevice));
  double start_time, total_time;
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr, true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr);
  init_array(sampler.result.length, sampler.result.size,
             sampler.result.hop_num);
  // allocate global buffer
  int block_num = FLAGS_sm;
  int gbuff_size = sampler.ggraph.MaxDegree;

  LOG("alllocate GMEM buffer %d MB\n",
      block_num * gbuff_size * MEM_PER_ELE / 1024 / 1024);

  Vector_pack<uint> *vector_pack_h = new Vector_pack<uint>[block_num];
  for (size_t i = 0; i < block_num; i++) {
    vector_pack_h[i].Allocate(gbuff_size, sampler.device_id);
  }
  CUDA_RT_CALL(cudaDeviceSynchronize());
  Vector_pack<uint> *vector_packs;
  CUDA_RT_CALL(
      cudaMalloc(&vector_packs, sizeof(Vector_pack<uint>) * block_num));
  CUDA_RT_CALL(cudaMemcpy(vector_packs, vector_pack_h,
                          sizeof(Vector_pack<uint>) * block_num,
                          cudaMemcpyHostToDevice));

  float *tp_d, tp;
  tp = FLAGS_tp;
  cudaMalloc(&tp_d, sizeof(float));
  CUDA_RT_CALL(cudaMemcpy(tp_d, &tp, sizeof(float), cudaMemcpyHostToDevice));
 



  Walker *sampler_ptr2;
  cudaMalloc(&sampler_ptr2, sizeof(Walker));
  CUDA_RT_CALL(cudaMemcpy(sampler_ptr2, &sampler2, sizeof(Walker),
                          cudaMemcpyHostToDevice));
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr2, true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr2);
  init_array(sampler2.result.length, sampler2.result.size,
             sampler2.result.hop_num);
  // allocate global buffer
  block_num = FLAGS_sm;
  gbuff_size = sampler2.ggraph.MaxDegree;
  
  LOG("alllocate GMEM buffer %d MB\n",
      block_num * gbuff_size * MEM_PER_ELE / 1024 / 1024);

  Vector_pack<uint> *vector_pack_h2 = new Vector_pack<uint>[block_num];
  for (size_t i = 0; i < block_num; i++) {
    vector_pack_h2[i].Allocate(gbuff_size, sampler.device_id);
  }
  CUDA_RT_CALL(cudaDeviceSynchronize());
  Vector_pack<uint> *vector_packs2;
  CUDA_RT_CALL(
      cudaMalloc(&vector_packs2, sizeof(Vector_pack<uint>) * block_num));
  CUDA_RT_CALL(cudaMemcpy(vector_packs2, vector_pack_h2,
                          sizeof(Vector_pack<uint>) * block_num,
                          cudaMemcpyHostToDevice));

  Walker *sampler_ptr3;
  cudaMalloc(&sampler_ptr3, sizeof(Walker));
  CUDA_RT_CALL(cudaMemcpy(sampler_ptr3, &sampler3, sizeof(Walker),
                          cudaMemcpyHostToDevice));
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr3, true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr3);
  init_array(sampler3.result.length, sampler3.result.size,
             sampler3.result.hop_num);
  // allocate global buffer
  block_num = FLAGS_sm;
  gbuff_size = sampler2.ggraph.MaxDegree;
  
  LOG("alllocate GMEM buffer %d MB\n",
      block_num * gbuff_size * MEM_PER_ELE / 1024 / 1024);

  Vector_pack<uint> *vector_pack_h3 = new Vector_pack<uint>[block_num];
  for (size_t i = 0; i < block_num; i++) {
    vector_pack_h3[i].Allocate(gbuff_size, sampler.device_id);
  }
  CUDA_RT_CALL(cudaDeviceSynchronize());
  Vector_pack<uint> *vector_packs3;
  CUDA_RT_CALL(
      cudaMalloc(&vector_packs3, sizeof(Vector_pack<uint>) * block_num));
  CUDA_RT_CALL(cudaMemcpy(vector_packs3, vector_pack_h3,
                          sizeof(Vector_pack<uint>) * block_num,
                          cudaMemcpyHostToDevice));


  Walker *sampler_ptr4;
  cudaMalloc(&sampler_ptr4, sizeof(Walker));
  CUDA_RT_CALL(cudaMemcpy(sampler_ptr4, &sampler4, sizeof(Walker),
                          cudaMemcpyHostToDevice));
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr3, true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr3);
  init_array(sampler4.result.length, sampler4.result.size,
             sampler4.result.hop_num);
  // allocate global buffer
  block_num = FLAGS_sm;
  gbuff_size = sampler4.ggraph.MaxDegree;
  
  LOG("alllocate GMEM buffer %d MB\n",
      block_num * gbuff_size * MEM_PER_ELE / 1024 / 1024);

  Vector_pack<uint> *vector_pack_h4 = new Vector_pack<uint>[block_num];
  for (size_t i = 0; i < block_num; i++) {
    vector_pack_h4[i].Allocate(gbuff_size, sampler.device_id);
  }
  CUDA_RT_CALL(cudaDeviceSynchronize());
  Vector_pack<uint> *vector_packs4;
  CUDA_RT_CALL(
      cudaMalloc(&vector_packs4, sizeof(Vector_pack<uint>) * block_num));
  CUDA_RT_CALL(cudaMemcpy(vector_packs4, vector_pack_h4,
                          sizeof(Vector_pack<uint>) * block_num,
                          cudaMemcpyHostToDevice));



  //  Global_buffer
  CUDA_RT_CALL(cudaDeviceSynchronize());
  start_time = wtime();
  if (FLAGS_static) {
    for (size_t i = 0; i < sampler3.result.hop_num - 1; i++) {
      OnlineWalkKernelStatic<<<block_num, BLOCK_SIZE, 0, 0>>>(
          sampler_ptr3, vector_packs3, i, tp_d, FLAGS_m);
      CUDA_RT_CALL(cudaDeviceSynchronize());
      CUDA_RT_CALL(cudaPeekAtLastError());
    }

  } else {
    if (FLAGS_debug)
      OnlineWalkKernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs,
                                                tp_d);
    else
      OnlineWalkKernel<<<block_num/2, BLOCK_SIZE, 0, stream1>>>(sampler_ptr,
                                                        vector_packs, tp_d);
      OnlineWalkKernel<<<block_num/2, BLOCK_SIZE, 0, stream2>>>(sampler_ptr2,
                                                        vector_packs2, tp_d);
      OnlineWalkKernel<<<block_num/2, BLOCK_SIZE, 0, stream2>>>(sampler_ptr3,
                                                        vector_packs3, tp_d);
      OnlineWalkKernel<<<block_num/2, BLOCK_SIZE, 0, stream2>>>(sampler_ptr4,
                                                        vector_packs4, tp_d);                                                  
      cout<<"blocks: "<<block_num<<endl;
  }

  CUDA_RT_CALL(cudaDeviceSynchronize());
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  LOG("Device %d sampling time:\t%.2f ms ratio:\t %.1f MSEPS\n",
      omp_get_thread_num(), total_time * 1000,
      static_cast<float>(sampler.result.GetSampledNumber() / total_time /
                         1000000));
  sampler2.sampled_edges = sampler.result.GetSampledNumber();
  LOG("sampled_edges %d\n", sampler.sampled_edges);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return total_time;
}






float Mixwalk(Walker &sampler, Walker &sampler2) {
  LOG("%s\n", __FUNCTION__);
#ifdef skip8k
  LOG("skipping 8k\n");
#endif  // skip8k

  LOG("overring staic flag, static\n");
  FLAGS_static=0;

  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  int n_sm = prop.multiProcessorCount;
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  Walker *sampler_ptr;
  cudaMalloc(&sampler_ptr, sizeof(Walker));
  CUDA_RT_CALL(cudaMemcpy(sampler_ptr, &sampler, sizeof(Walker),
                          cudaMemcpyHostToDevice));
  double start_time, total_time;
  init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr, true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr);
  init_array(sampler.result.length, sampler.result.size,
             sampler.result.hop_num);
  // allocate global buffer
  int block_num = FLAGS_sm;
  int gbuff_size = sampler.ggraph.MaxDegree;
  ;
  LOG("alllocate GMEM buffer %d MB\n",
      block_num * gbuff_size * MEM_PER_ELE / 1024 / 1024);

  Vector_pack<uint> *vector_pack_h = new Vector_pack<uint>[block_num];
  for (size_t i = 0; i < block_num; i++) {
    vector_pack_h[i].Allocate(gbuff_size, sampler.device_id);
  }
  CUDA_RT_CALL(cudaDeviceSynchronize());
  Vector_pack<uint> *vector_packs;
  CUDA_RT_CALL(
      cudaMalloc(&vector_packs, sizeof(Vector_pack<uint>) * block_num));
  CUDA_RT_CALL(cudaMemcpy(vector_packs, vector_pack_h,
                          sizeof(Vector_pack<uint>) * block_num,
                          cudaMemcpyHostToDevice));

  float *tp_d, tp;
  tp = FLAGS_tp;
  cudaMalloc(&tp_d, sizeof(float));
  CUDA_RT_CALL(cudaMemcpy(tp_d, &tp, sizeof(float), cudaMemcpyHostToDevice));

   
  Walker *sampler_ptr2;
  cudaMalloc(&sampler_ptr2, sizeof(Walker));
  CUDA_RT_CALL(
      cudaMemcpyAsync(sampler_ptr2, &sampler2, sizeof(Walker), cudaMemcpyHostToDevice,stream2));
  // init_kernel_ptr<<<1, 32, 0, 0>>>(sampler_ptr,true);
  BindResultKernel<<<1, 32, 0, 0>>>(sampler_ptr2);

  //  Global_buffer
  CUDA_RT_CALL(cudaDeviceSynchronize());
  start_time = wtime();
  if (FLAGS_static) {
    for (size_t i = 0; i < sampler.result.hop_num - 1; i++) {
      OnlineWalkKernelStatic<<<block_num, BLOCK_SIZE, 0, 0>>>(
          sampler_ptr, vector_packs, i, tp_d, FLAGS_m);
      CUDA_RT_CALL(cudaDeviceSynchronize());
      CUDA_RT_CALL(cudaPeekAtLastError());
    }

  } else {
    if (FLAGS_debug)
      OnlineWalkKernel<<<1, BLOCK_SIZE, 0, 0>>>(sampler_ptr, vector_packs,
                                                tp_d);
    else{
      OnlineWalkKernel<<<block_num, BLOCK_SIZE, 0, stream1>>>(sampler_ptr,
                                                        vector_packs, tp_d);
      sample_kernel<<<block_num/2, BLOCK_SIZE, 0, stream2>>>(sampler_ptr2);
                                                        
     
     CUDA_RT_CALL(cudaDeviceSynchronize());
     cout<<"blocks used in mix kernel: "<<block_num<<endl;
     cout<<"online"<<endl;
     print_result<<<1, 32, 0, 0>>>(sampler_ptr);
     cout<<"sample"<<endl;
     print_result<<<1, 32, 0, 0>>>(sampler_ptr2);
    }
      
  }

  CUDA_RT_CALL(cudaDeviceSynchronize());
  // CUDA_RT_CALL(cudaPeekAtLastError());
  total_time = wtime() - start_time;
  LOG("Device %d sampling time:\t%.2f ms ratio:\t %.1f MSEPS\n",
      omp_get_thread_num(), total_time * 1000,
      static_cast<float>(sampler.result.GetSampledNumber() / total_time /
                         1000000));
  sampler.sampled_edges = sampler.result.GetSampledNumber();
  LOG("sampled_edges %d\n", sampler.sampled_edges);
  if (FLAGS_printresult) print_result<<<1, 32, 0, 0>>>(sampler_ptr);
  CUDA_RT_CALL(cudaDeviceSynchronize());
  return total_time;
}