/*
 * @Description:
 * @Date: 2020-11-17 13:28:27
 * @LastEditors: Pengyu Wang
 * @LastEditTime: 2021-01-15 14:32:23
 * @FilePath: /skywalker/src/main.cu
 */
#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <numa.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>

#include "gpu_graph.cuh"
#include "graph.cuh"
#include "sampler.cuh"
#include "sampler_result.cuh"
using namespace std;
// DECLARE_bool(v);
// DEFINE_bool(pf, false, "use UM prefetch");
DEFINE_string(input, "/home/pywang/data/lj.w.gr", "input");
DEFINE_string(input2, "/home/pywang/data/lj.w.gr", "input2");
DEFINE_string(input3, "/home/pywang/data/lj.w.gr", "input3");
DEFINE_string(inputalias, "/home/xucheng/alias", "directly load alias table");
DEFINE_string(savealias, "lj", "save alias table");
DEFINE_string(readalias, "lj", "read alias table");
// DEFINE_int32(device, 0, "GPU ID");
DEFINE_int32(ngpu, 1, "number of GPUs ");
DEFINE_bool(s, true, "single gpu");
DEFINE_bool(load, false, "load alias table");
DEFINE_bool(k1, false, "only run k1");
DEFINE_bool(k2, false, "only run k2");
DEFINE_bool(save, false, "save alias table");
DEFINE_bool(direct, false, "directly use alias table");
DEFINE_bool(cnode, false, "concurrent node2vec");
DEFINE_bool(mix, false, "offline+online");
DEFINE_int32(n, 4000, "sample size");
DEFINE_int32(k, 2, "neightbor");
DEFINE_int32(d, 2, "depth");

DEFINE_double(hd, 1, "high degree ratio");
DEFINE_bool(concurrent, false, "run two kernels");
DEFINE_int32(level, 1, "the number of concurrent kernels");
DEFINE_bool(ol, false, "online alias table building");
DEFINE_bool(rw, false, "Random walk specific");
DEFINE_bool(mul, false, "multiple gpu graphs");
DEFINE_bool(dw, false, "using degree as weight");
DEFINE_bool(cp, false, "copy to pinned memory");
DEFINE_bool(zcp, false, "using zero copy");
DEFINE_bool(samegraph, false, "multiple tasks on the same graph");

DEFINE_bool(randomweight, false, "generate random weight with range");
DEFINE_int32(weightrange, 2, "generate random weight with range from 0 to ");

// app specific
DEFINE_bool(sage, false, "GraphSage");
DEFINE_bool(deepwalk, false, "deepwalk");
DEFINE_bool(node2vec, false, "node2vec");
DEFINE_bool(ppr, false, "ppr");
DEFINE_double(p, 2.0, "hyper-parameter p for node2vec");
DEFINE_double(q, 0.5, "hyper-parameter q for node2vec");
DEFINE_double(tp, 0.0, "terminate probabiility");

DEFINE_bool(hmtable, false, "using host mapped mem for alias table");
DEFINE_bool(dt, true, "using duplicated table on each GPU");

DEFINE_bool(umgraph, true, "using UM for graph");
DEFINE_bool(hmgraph, false, "using host registered mem for graph");
DEFINE_bool(gmgraph, false, "using GPU mem for graph");
DEFINE_int32(gmid, 1, "using mem of GPU gmid for graph");

DEFINE_bool(umtable, false, "using UM for alias table");
DEFINE_bool(umresult, false, "using UM for result");
DEFINE_bool(umbuf, false, "using UM for global buffer");

DEFINE_bool(cache, false, "cache alias table for online");
DEFINE_bool(debug, false, "debug");
DEFINE_bool(bias, true, "biased or unbiased sampling");
DEFINE_bool(full, false, "sample over all node");
DEFINE_bool(stream, false, "streaming sample over all node");

DEFINE_bool(v, false, "verbose");
DEFINE_bool(printresult, false, "printresult");

DEFINE_bool(edgecut, true, "edgecut");

DEFINE_bool(itl, true, "interleave");
DEFINE_bool(twc, true, "using twc");
DEFINE_bool(static, true, "using static scheduling");
DEFINE_bool(buffer, true, "buffered write for memory");

DEFINE_int32(m, 4, "block per sm");
DEFINE_int32(sm, 68, "number of sm");
DEFINE_int32(energy, 250, "energy limit");
DEFINE_int32(sf, 1, "static factor");
DEFINE_bool(peritr, false, "invoke kernel for each itr");

DEFINE_bool(sp, false, "using spliced buffer");

DEFINE_bool(pf, true, "using UM prefetching");
DEFINE_bool(ab, true, "using UM AB hint");
// DEFINE_bool(pf, true, "using UM prefetching");

DEFINE_bool(async, false, "using async execution");
DEFINE_bool(replica, false, "same task for all gpus");
DEFINE_bool(built, false, "has built table");
DEFINE_bool(trans, false, "transfer from global table");
DEFINE_bool(gmem, false, "do not use shmem as buffer");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (numa_available() < 0) {
    LOG("Your system does not support NUMA API\n");
  }
  // cout << "ELE_PER_BLOCK " << ELE_PER_BLOCK << " ELE_PER_WARP " <<
  // ELE_PER_WARP
  //      << "ALLOWED_ELE_PER_SUBWARP " << ALLOWED_ELE_PER_SUBWARP << endl;

  // override flag
  if (FLAGS_hmgraph) {
    FLAGS_umgraph = false;
    FLAGS_gmgraph = false;
  }
  if (FLAGS_gmgraph) {
    FLAGS_umgraph = false;
    FLAGS_hmgraph = false;

    int can_access_peer_0_1;
    CUDA_RT_CALL(cudaDeviceCanAccessPeer(&can_access_peer_0_1, 0, FLAGS_gmid));
    //   if (can_access_peer_0_1 == 0) {
    //     printf("no p2p\n");
    //     return 1;
    //   }
  }
  if (FLAGS_node2vec) {
    // FLAGS_ol = true;
    // FLAGS_bias = false;  //we could run node2vec in unbiased app currently.
    FLAGS_rw = true;
    FLAGS_k = 1;
    FLAGS_d = 100;
  }
  if (FLAGS_deepwalk) {
    // FLAGS_ol=true;
    FLAGS_rw = true;
    FLAGS_k = 1;
    FLAGS_d = 100;
  }
  if (FLAGS_ppr) {
    // FLAGS_ol=true;
    FLAGS_rw = true;
    FLAGS_k = 1;
    FLAGS_d = 100;
    FLAGS_tp = 0.15;
  }
  if (FLAGS_sage) {
    // FLAGS_ol=true;
    FLAGS_rw = false;
    FLAGS_d = 2;
  }
  if (!FLAGS_bias) {
    FLAGS_weight = false;
    FLAGS_randomweight = false;
  }
#ifdef SPEC_EXE
  LOG("SPEC_EXE \n");
#endif

  int sample_size = FLAGS_n;
  int NeighborSize = FLAGS_k;
  int Depth = FLAGS_d;

  // uint hops[3]{1, 2, 2};
  uint *hops = new uint[Depth + 1];
  hops[0] = 1;
  for (size_t i = 1; i < Depth + 1; i++) {
    hops[i] = NeighborSize;
  }
  if (FLAGS_sage) {
    hops[1] = 25;
    hops[2] = 10;
  }
  Graph *ginst = new Graph(FLAGS_input);
  Graph *ginst2 = new Graph(FLAGS_input2);
  // Graph *ginst3 = new Graph(FLAGS_input3);

  if (ginst->numEdge > 600000000 || ginst2->numEdge > 600000000) {
    FLAGS_umtable = 1;
    LOG("overriding um for alias table\n");
  }
  // FLAGS_umtable=0;
  if (ginst->MaxDegree > 500000) {
    FLAGS_umbuf = 1;
    LOG("overriding um buffer\n");
  }
  if (FLAGS_full && !FLAGS_stream) {
    sample_size = ginst->numNode;
    FLAGS_n = ginst->numNode;
  }

  // uint num_device = FLAGS_ngpu;
  float *times = new float[FLAGS_ngpu];
  double start, end;
  float *tp = new float[FLAGS_ngpu];
  float *table_times = new float[FLAGS_ngpu];
  for (size_t num_device = 1; num_device < FLAGS_ngpu + 1; num_device++) {
    if (FLAGS_s) num_device = FLAGS_ngpu;
    AliasTable global_table;
    if (num_device > 1 && !FLAGS_ol) {
      global_table.Alocate(ginst->numNode, ginst->numEdge);
    }
    if (FLAGS_trans) {
      global_table.Alocate(ginst->numNode, ginst->numEdge);
    }
    gpu_graph *ggraphs = new gpu_graph[num_device];
    gpu_graph *ggraphs2 = new gpu_graph[num_device];
    Sampler *samplers2 = new Sampler[num_device];
    // gpu_graph *ggraphs3 = new gpu_graph[num_device];
    // Sampler *samplers3 = new Sampler[num_device];

    Sampler *samplers = new Sampler[num_device];

    Sampler_new *samplers_new = new Sampler_new[num_device];
    float time[num_device];
    float time2[num_device];
    //   float time3[num_device];

#pragma omp parallel num_threads(num_device) \
    shared(ginst, ggraphs, samplers, global_table, samplers_new)
    {
      int dev_id = omp_get_thread_num();
      int dev_num = omp_get_num_threads();
      uint local_sample_size = sample_size / dev_num;
      if (FLAGS_replica) local_sample_size = sample_size;

      // if (dev_id < 2) {
      //   numa_run_on_on_node(0);
      //   numa_set_prefered(0);
      // } else {
      //   numa_run_on_on_node(1);
      //   numa_set_prefered(1);
      // }

      LOG("device_id %d ompid %d coreid %d\n", dev_id, omp_get_thread_num(),
          sched_getcpu());
      CUDA_RT_CALL(cudaSetDevice(dev_id));
      CUDA_RT_CALL(cudaFree(0));
      // FLAGS_gmgraph=1;
      // FLAGS_umgraph=0;
      ggraphs[dev_id] = gpu_graph(ginst, dev_id);
      samplers[dev_id] = Sampler(ggraphs[dev_id], dev_id);
      if (FLAGS_load == 0 && FLAGS_bias == 1 && FLAGS_ol == 0) {
        samplers[dev_id].InitFullForConstruction(dev_num, dev_id);
        time[dev_id] = ConstructTable(samplers[dev_id], dev_num, dev_id);
      }
      if (FLAGS_load == 1) {
        cout << "loading alias table" << endl;
        string aliasfile = "./alias/" + FLAGS_readalias + "-alias.txt";
        string probfile = "./alias/" + FLAGS_readalias + "-prob.txt";
        string validfile = "./alias/" + FLAGS_readalias + "-valid.txt";
        ifstream fin, fin1, fin2;
        int size = ginst->numEdge;
        fin.open(probfile.c_str(), ios::binary);
        float *ptrh21 = new float[size];
        fin.read((char *)ptrh21, size * sizeof(float));

        // for (int i=0; i<100;i++){
        //   cout<<ptrh21[i]<<endl;
        // }
        //    delete[] ptrh21;
        fin.close();

        fin1.open(aliasfile.c_str(), ios::binary);
        uint *ptrh22 = new uint[size];

        fin1.read((char *)ptrh22, size * sizeof(uint));

        // for (int i=0; i<100;i++){
        //   cout<<ptrh22[i]<<endl;
        // }
        //   delete[] ptrh22;
        fin1.close();
        size = ginst->numNode;
        fin2.open(validfile.c_str(), ios::binary);
        char *ptrh23 = new char[size];
        fin2.read((char *)ptrh23, size * sizeof(char));
        // for (int i=0; i<100;i++){
        //   cout<<ptrh23[i]<<endl;
        // }
        //  delete[] ptrh23;
        fin2.close();
        start = wtime();

        if (FLAGS_cp) {
          float *prob;
          uint *alias;
          char *valid;
          //  CUDA_RT_CALL(cudaHostAlloc((void **)&samplers[dev_id].prob_array,
          //                      samplers[dev_id].ggraph.local_edge_num *
          //                      sizeof(float), cudaHostAllocWriteCombined));
          //  CUDA_RT_CALL(cudaHostAlloc((void **)&samplers[dev_id].alias_array,
          //                      samplers[dev_id].ggraph.local_edge_num *
          //                      sizeof(uint), cudaHostAllocWriteCombined));
          //  CUDA_RT_CALL(cudaHostAlloc((void **)&samplers[dev_id].valid,
          //                      samplers[dev_id].ggraph.local_vtx_num *
          //                      sizeof(char), cudaHostAllocWriteCombined));

          // CUDA_RT_CALL(cudaMemcpy(samplers[dev_id].prob_array,ptrh21,samplers[dev_id].ggraph.local_edge_num
          // * sizeof(float),cudaMemcpyHostToDevice));
          // CUDA_RT_CALL(cudaMemcpy(samplers[dev_id].alias_array,ptrh22,samplers[dev_id].ggraph.local_edge_num
          // * sizeof(uint),cudaMemcpyHostToDevice));
          // CUDA_RT_CALL(cudaMemcpy(samplers[dev_id].valid,ptrh23,samplers[dev_id].ggraph.local_vtx_num
          // * sizeof(char),cudaMemcpyHostToDevice));

          CUDA_RT_CALL(cudaMallocManaged((void **)&prob,
                                         ginst->numEdge * sizeof(float)));
          CUDA_RT_CALL(cudaMallocManaged((void **)&alias,
                                         ginst->numEdge * sizeof(uint)));
          CUDA_RT_CALL(cudaMallocManaged((void **)&valid,
                                         ginst->numNode * sizeof(char)));

          //  cout<<"aaaaa"<<endl;
          CUDA_RT_CALL(cudaMemcpy(prob, ptrh21, ginst->numEdge * sizeof(float),
                                  cudaMemcpyHostToDevice));
          CUDA_RT_CALL(cudaMemcpy(alias, ptrh22, ginst->numEdge * sizeof(uint),
                                  cudaMemcpyHostToDevice));
          CUDA_RT_CALL(cudaMemcpy(valid, ptrh23, ginst->numNode * sizeof(char),
                                  cudaMemcpyHostToDevice));

          end = wtime();
          float *testt = new float[ginst->numEdge];
          CUDA_RT_CALL(cudaMemcpy(testt, prob, ginst->numEdge * sizeof(float),
                                  cudaMemcpyDeviceToHost));

          // for (int i=0; i<100;i++){
          //   cout<<"prob:"<<testt[i]<<endl;
          // }
          delete[] testt;
          // for (int i=0; i<100;i++){
          //   cout<<alias[i]<<endl;
          // }

          cout << "direct cpy time:" << end - start << endl;

          // CUDA_RT_CALL(cudaFreeHost(samplers[dev_id].prob_array));
          // CUDA_RT_CALL(cudaFreeHost(samplers[dev_id].alias_array));
          // CUDA_RT_CALL(cudaFreeHost(samplers[dev_id].valid));

          // CUDA_RT_CALL(cudaFree(prob));
          // CUDA_RT_CALL(cudaFree(alias));
          // CUDA_RT_CALL(cudaFree(valid));
          CUDA_RT_CALL(cudaDeviceSynchronize());

          CUDA_RT_CALL(cudaFree(prob));
          CUDA_RT_CALL(cudaFree(alias));
          CUDA_RT_CALL(cudaFree(valid));
        }

        if (FLAGS_zcp) {
          // for (int i=0; i<100;i++){
          //   cout<<"prob: "<<ptrh21[i]<<endl;
          // }
          start = wtime();
          cudaHostRegister(ptrh21, ginst->numEdge * sizeof(float),
                           cudaHostRegisterDefault);
          cudaHostRegister(ptrh22, ginst->numEdge * sizeof(uint),
                           cudaHostRegisterDefault);
          cudaHostRegister(ptrh23, ginst->numNode * sizeof(char),
                           cudaHostRegisterDefault);
          //  for (int i=0; i<100;i++){
          //   cout<<"register: "<<ptrh21[i]<<endl;
          // }

          CUDA_RT_CALL(cudaHostGetDevicePointer(
              (void **)&samplers[dev_id].prob_array, (void *)ptrh21, 0));
          CUDA_RT_CALL(cudaHostGetDevicePointer(
              (void **)&samplers[dev_id].alias_array, (void *)ptrh22, 0));
          CUDA_RT_CALL(cudaHostGetDevicePointer(
              (void **)&samplers[dev_id].valid, (void *)ptrh23, 0));
          end = wtime();
          cout << "zero-copy time: " << end - start << endl;
          // for (int i=0; i<100;i++){
          //   cout << "deviceData = " << samplers[dev_id].prob_array[i] <<
          //   std::endl;
          // }
        }

        // delete[] ptrh21;
        // delete[] ptrh22;
        // delete[] ptrh23;
        if (FLAGS_bias == 1 ) {
          samplers[dev_id].InitFullForConstruction(dev_num, dev_id);
          time[dev_id] = ConstructTable(samplers[dev_id], dev_num, dev_id);
        }
      }
   //   FLAGS_load = 0;
      FLAGS_built = 0;
      //  FLAGS_gmgraph=1;
      //  FLAGS_gmgraph=0;
      // FLAGS_hmgraph=1;
      // FLAGS_umgraph=1;
      if (FLAGS_concurrent || FLAGS_mix || FLAGS_cnode && !FLAGS_samegraph) {
        ggraphs2[dev_id] = gpu_graph(ginst2, dev_id);
        samplers2[dev_id] = Sampler(ggraphs2[dev_id], dev_id);
        //   ggraphs3[dev_id]= gpu_graph(ginst3,dev_id);
        //    samplers3[dev_id] = Sampler(ggraphs3[dev_id], dev_id);
      }
      if (FLAGS_samegraph) {
        ggraphs2[dev_id] = gpu_graph(ginst2, ggraphs[dev_id], dev_id);
        cout << "pointer:" << ggraphs[dev_id].xadj << endl;
        cout << "pointer:" << ggraphs2[dev_id].xadj << endl;
        samplers2[dev_id] = Sampler(ggraphs[dev_id], dev_id);
        //   ggraphs3[dev_id]= gpu_graph(ginst3,dev_id);
        //    samplers3[dev_id] = Sampler(ggraphs3[dev_id], dev_id);
      }
      // FLAGS_gmgraph=1;
      // FLAGS_hmgraph=0;

      if (!FLAGS_bias) {
        if (FLAGS_rw) {
          Walker walker(samplers[dev_id]);
          walker.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
#pragma omp barrier
          time[dev_id] = UnbiasedWalk(walker);
          samplers[dev_id].sampled_edges = walker.sampled_edges;
        } else {
          samplers[dev_id].SetSeed(local_sample_size, Depth + 1, hops, dev_num,
                                   dev_id);

          samplers_new[dev_id] = samplers[dev_id];
          time[dev_id] = UnbiasedSample(samplers_new[dev_id]);
        }
      }

      if (FLAGS_bias && FLAGS_ol) {  // online biased
        samplers[dev_id].SetSeed(local_sample_size, Depth + 1, hops, dev_num,
                                 dev_id);
        if (!FLAGS_rw) {
          // if (!FLAGS_sp)
          if (!FLAGS_twc)
            time[dev_id] = OnlineGBSample(samplers[dev_id]);
          else
            time[dev_id] = OnlineGBSampleTWC(samplers[dev_id]);
          // else
          // time[dev_id] = OnlineSplicedSample(samplers[dev_id]); //to add
          // spliced
        } else {
          Walker walker(samplers[dev_id]);
          walker.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
          // Walker walker2(samplers2[dev_id]);
          // walker2.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
          if (FLAGS_gmem) {
            time[dev_id] = OnlineWalkGMem(walker);
          } else {
            if (FLAGS_cnode == 0 && FLAGS_mix == 0) {
              time[dev_id] = OnlineWalkShMem(walker);

            } else {
              // samplers2[dev_id].InitFullForConstruction(dev_num, dev_id);
              // time2[dev_id]= ConstructTable(samplers2[dev_id], dev_num,
              // dev_id); Walker walker2(samplers2[dev_id]);
              // walker2.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
              // start=wtime();
              // time[dev_id] = OfflineWalk(walker2);
              // time[dev_id] = OnlineWalkShMem(walker);
              // CUDA_RT_CALL(cudaDeviceSynchronize());
              // end=wtime();
              // cout<<"Mix Time:" <<end-start<<endl;
              if (FLAGS_mix == 1) {
                samplers2[dev_id].InitFullForConstruction(dev_num, dev_id);
                time2[dev_id] =
                    ConstructTable(samplers2[dev_id], dev_num, dev_id);
                Walker walker2(samplers2[dev_id]);
                walker2.SetSeed(local_sample_size * 10, Depth + 1, dev_num,
                                dev_id);
                // Mixwalk(walker, walker2);
                start = wtime();
                time[dev_id] = Mixwalk(walker, walker2);
                end = wtime();
                cout << "include memory cpy:" << end - start << endl;
              }
              if (FLAGS_cnode == 1 && FLAGS_level == 1) {
                Walker walker2(samplers2[dev_id]);
                walker2.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);

                start = wtime();
                time[dev_id] = OnlineWalkShMem2(walker, walker2);
                end = wtime();
                cout << "include memory cpy:" << end - start << endl;
              }
              if (FLAGS_cnode == 1 && FLAGS_level == 4) {
                Walker walker2(samplers2[dev_id]);
                walker2.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
                Walker walker3(samplers2[dev_id]);
                walker3.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
                Walker walker4(samplers2[dev_id]);
                walker4.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
                start = wtime();
                time[dev_id] =
                    OnlineWalkShMem4(walker, walker2, walker3, walker4);
                end = wtime();
                cout << "include memory cpy:" << end - start << endl;
              }
            }
          }

          samplers[dev_id].sampled_edges = walker.sampled_edges;
        }
      }

      if (FLAGS_bias && !FLAGS_ol) {  // offline biased
        FLAGS_load=0;
        //        samplers[dev_id].InitFullForConstruction(dev_num, dev_id);
        if (FLAGS_concurrent) {
          cout<<"concurrent1"<<endl;
        //  FLAGS_umtable = 0;
          samplers2[dev_id].InitFullForConstruction(dev_num, dev_id);
          //        samplers3[dev_id].InitFullForConstruction(dev_num, dev_id);
        }
     
          //       time[dev_id] = ConstructTable(samplers[dev_id], dev_num,
          //       dev_id);
          //        FLAGS_umtable=1;
          if (FLAGS_concurrent) {
             cout<<"concurrent2"<<endl;
            time2[dev_id] = ConstructTable(samplers2[dev_id], dev_num, dev_id);
            //        time3[dev_id]= ConstructTable(samplers3[dev_id], dev_num,
            //        dev_id);
          }
          //        FLAGS_umtable=0;
        

        if (FLAGS_save) {
          ofstream fout, fout1, fout2;
          string aliasfile = "./alias/" + FLAGS_savealias + "-alias.txt";
          string probfile = "./alias/" + FLAGS_savealias + "-prob.txt";
          string validfile = "./alias/" + FLAGS_savealias + "-valid.txt";
          //   cout<<address<<endl;
          fout.open(probfile.c_str(), ios::binary);
          //  printH(samplers[dev_id].prob_array,100);
          int size = ginst->numEdge;
          float *ptrh = new float[size];
          CUDA_RT_CALL(cudaMemcpy(ptrh, samplers[dev_id].prob_array,
                                  size * sizeof(float), cudaMemcpyDefault));
          fout.write((const char *)ptrh, size * sizeof(float));
          // for (size_t i = 0; i < size; i++) {
          //     // printf("%d\t", ptrh[i]);
          //    fout<< ptrh[i] << endl;
          // }
          delete[] ptrh;
          fout.close();
          fout1.open(aliasfile.c_str(), ios::binary);
          //  printH(samplers[dev_id].prob_array,100);
          size = ginst->numEdge;
          uint *ptrh1 = new uint[size];
          CUDA_RT_CALL(cudaMemcpy(ptrh1, samplers[dev_id].alias_array,
                                  size * sizeof(uint), cudaMemcpyDefault));
          fout1.write((const char *)ptrh1, size * sizeof(uint));
          // for (size_t i = 0; i < size; i++) {
          //     // printf("%d\t", ptrh[i]);
          //    fout<< ptrh[i] << endl;
          // }
          delete[] ptrh1;
          fout1.close();
          fout2.open(validfile.c_str(), ios::binary);
          //  printH(samplers[dev_id].prob_array,100);
          size = ginst->numNode;
          char *ptrh11 = new char[size];
          CUDA_RT_CALL(cudaMemcpy(ptrh11, samplers[dev_id].alias_array,
                                  size * sizeof(char), cudaMemcpyDefault));
          fout2.write((const char *)ptrh11, size * sizeof(char));
          // for (size_t i = 0; i < size; i++) {
          //     // printf("%d\t", ptrh[i]);
          //    fout<< ptrh[i] << endl;
          // }
          delete[] ptrh11;
          fout2.close();

          ifstream fin;
          size = ginst->numEdge;
          fin.open(probfile.c_str(), ios::binary);
          float *ptrh2 = new float[size];
          start = wtime();
          fin.read((char *)ptrh2, size * sizeof(float));
          end = wtime();
          for (int i = 0; i < 100; i++) {
            cout << ptrh2[i] << endl;
          }
          delete[] ptrh2;
          cout << "Read table time:" << end - start << endl;

          //   //      for (int i=0; i<ginst->numEdge; i++){
          //           fout<<(void *)(samplers[dev_id].prob_array)<<endl;
          //  //       }
          //         cout<<"b"<<endl;
          //         fout.close();

          //         fout1.open(aliasfile.c_str());
          //         for (int i=0; i<ginst->numEdge; i++){
          //           fout1<<(void *)(samplers[dev_id].alias_array+i)<<endl;
          //         }
          //         fout1.close();

          //         fout2.open(validfile.c_str());
          //         for (int i=0; i<ginst->numNode; i++){
          //           fout2<<(void *)(samplers[dev_id].valid+i)<<endl;
          //         }
          //         fout2.close();
        }
        // if (FLAGS_trans){
        //  //  global_table.Assemble(samplers[dev_id].ggraph);
        //    if (!FLAGS_dt)
        //     samplers[dev_id].UseGlobalAliasTable(global_table);
        //    else {
        //     LOG("CopyFromGlobalAliasTable\n");
        //     start= wtime();
        //     samplers[dev_id].CopyFromGlobalAliasTable(global_table);
        //     end=wtime();
        //     LOG("transfer table time:\t%.6f\n", end-start);
        //   }
        // }

        // use a global host mapped table for all gpus
        if (dev_num > 1 && FLAGS_n > 0) {
          global_table.Assemble(samplers[dev_id].ggraph);
          if (!FLAGS_dt)
            samplers[dev_id].UseGlobalAliasTable(global_table);
          else {
            //      LOG("CopyFromGlobalAliasTable\n");
            //      start= wtime();
            samplers[dev_id].CopyFromGlobalAliasTable(global_table);
            //     end=wtime();
            //      LOG("transfer table time:\t%.6f\n", end-start);
          }
        }
#pragma omp barrier
#pragma omp master
        {
          if (num_device > 1 && !FLAGS_ol && FLAGS_dt) {
            LOG("free global_table\n");
            global_table.Free();
          }

          LOG("Max construction time with %u gpu \t%.2f ms\n", dev_num,
              *max_element(time, time + num_device) * 1000);
          table_times[dev_num - 1] =
              *max_element(time, time + num_device) * 1000;

          FLAGS_built = true;
        }

        if (!FLAGS_rw) {  //&& FLAGS_k != 1
          samplers[dev_id].SetSeed(local_sample_size, Depth + 1, hops, dev_num,
                                   dev_id);
          samplers_new[dev_id] = samplers[dev_id];
          start = wtime();
          time[dev_id] = OfflineSample(samplers_new[dev_id]);
          end = wtime();
          cout << "include memory cpy:" << end - start << endl;
          // else
          //   time[dev_id] = AsyncOfflineSample(samplers[dev_id]);
        } else {
          Walker walker(samplers[dev_id]);
          walker.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
          if (FLAGS_concurrent) {
            if (FLAGS_level == 1) {
              Walker walker2(samplers2[dev_id]);
              walker2.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
              start = wtime();
              time[dev_id] = OfflineWalk2(walker, walker2);
              end = wtime();
              cout << "include memory cpy:" << end - start << endl;
            }
            if (FLAGS_level == 2) {
              Walker walker2(samplers2[dev_id]);
              Walker walker3(samplers2[dev_id]);
              walker2.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
              walker3.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
              start = wtime();
              time[dev_id] = OfflineWalk3(walker, walker2, walker3);
              end = wtime();
              cout << "include memory cpy:" << end - start << endl;
            }
            if (FLAGS_level == 3) {
              Walker walker2(samplers2[dev_id]);
              Walker walker3(samplers2[dev_id]);
              Walker walker4(samplers2[dev_id]);
              walker2.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
              walker3.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
              walker4.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
              start = wtime();
              time[dev_id] = OfflineWalk4(walker, walker2, walker3, walker4);
              end = wtime();
              cout << "include memory cpy:" << end - start << endl;
            }
            if (FLAGS_level == 4) {
              Walker walker2(samplers2[dev_id]);
              Walker walker3(samplers2[dev_id]);
              Walker walker4(samplers2[dev_id]);
              Walker walker5(samplers2[dev_id]);
              walker2.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
              walker3.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
              walker4.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
              walker5.SetSeed(local_sample_size, Depth + 1, dev_num, dev_id);
              start = wtime();
              time[dev_id] =
                  OfflineWalk5(walker, walker2, walker3, walker4, walker5);
              end = wtime();
              cout << "include memory cpy:" << end - start << endl;
            }
          } else {
            start = wtime();
            time[dev_id] = OfflineWalk(walker);
            end = wtime();
            cout << "include memory cpy:" << end - start << endl;
          }
          samplers[dev_id].sampled_edges = walker.sampled_edges;
          // if (FLAGS_concurrent){
          // samplers2[dev_id].sampled_edges = walker2.sampled_edges;
          // }
        }
        // if (dev_num == 1) {
        //    FLAGS_hmtable=0;
        samplers[dev_id].Free(dev_num == 1 ? false : true);
        // }
#pragma omp master
        {
          //
          if (num_device > 1 && !FLAGS_ol && !FLAGS_dt) {
            LOG("free global_table\n");
            global_table.Free();
          }
        }
      }
      ggraphs[dev_id].Free();
    }
    {
      size_t sampled = 0;
      if ((!FLAGS_bias || !FLAGS_ol) && (!FLAGS_rw))
        for (size_t i = 0; i < num_device; i++) {
          sampled += samplers_new[i].sampled_edges;  // / total_time /1000000
        }
      else
        for (size_t i = 0; i < num_device; i++) {
          sampled += samplers[i].sampled_edges;  // / total_time /1000000
        }
      float max_time = *max_element(time, time + num_device);
      // printf("%u GPU, %.2f ,  %.1f \n", num_device, max_time * 1000,
      //        sampled / max_time / 1000000);
      // printf("Max time %.5f ms with %u GPU, average TP %f MSEPS\n",
      //        max_time * 1000, num_device, sampled / max_time / 1000000);
      times[num_device - 1] = max_time * 1000;
      tp[num_device - 1] = sampled / max_time / 1000000;
    }
    if (FLAGS_s) break;
  }
  if (!FLAGS_ol && FLAGS_bias)
    for (size_t i = 0; i < FLAGS_ngpu; i++) {
      printf("%0.2f\t", table_times[i]);
    }
  printf("\n");
  for (size_t i = 0; i < FLAGS_ngpu; i++) {
    printf("%0.2f\t", times[i]);
  }
  printf("\n");
  for (size_t i = 0; i < FLAGS_ngpu; i++) {
    printf("%0.2f\t", tp[i]);
  }
  printf("\n");
  return 0;
}