#include <iostream>
#include <queue>
#include <limits>

#include "gswitch.h"

using G = device_graph_t<CSR, int>;
//const int maxi = std::numeric_limits<int>::max();

// actors
inspector_t inspector; 
selector_t selector;
executor_t executor;
feature_t fets;
config_t conf;
stat_t stats;

struct KCORE:Functor<VC, int, int, int>{
  __device__ Status filter(int vid, G g){

    int k = g.get_level() + 1;
    int* degree = wa_of(vid);
    int* core = ra_of(vid);
    if ((*degree) > k)
    {
        return Inactive;
    }
    else if ((*core) != -1)
    {
        return Fixed;
    }
    else
    {
        (*core) = k;
        return Active;
    }
  }
  __device__ int emit(int vid, int* w, G g){
    // return g.get_level() + 1;
    return 0;
  }

  __device__ bool comp(int* disu, int newv, G g){

    int k =  g.get_level() + 1;
    int old_degrees = (*disu);
    (*disu)--;
    // int old_degrees = atomicAdd(disu, -1);
    // if (old_degrees == k+1)
    // {
    //     return true;
    // }
    return false;
  }

  __device__ bool compAtomic(int* v, int newv, G g){
    
    int k = g.get_level() + 1;
    int old_degrees = atomicAdd(v, -1);
    // if (old_degrees == k+1)
    // {
    //     return true;
    // }
    return false;
  }

  // true means there's work to do
  __device__ bool exit(int v, G g) { 
    int k = g.get_level() + 1;
    return *wa_of(v) <= k && *ra_of(v) == -1;
  }
};

// int* spfa(host_graph_t<CSR,int> hg, int root){
//   std::cout << "generate CPU SSSP reference" << std::endl;
//   double ms = mwtime();
//   int* dis = (int*)malloc(sizeof(int)*hg.nvertexs);
//   int* vis = (int*)malloc(sizeof(int)*hg.nvertexs);
//   memset(vis,0,sizeof(int)*hg.nvertexs);
//   std::queue<int> q;
//   for(int i=0;i<hg.nvertexs;i++) dis[i]=maxi;
//   dis[root] = 0;
//   vis[root] = 1;
//   q.push(root);
//   while(!q.empty()){
//     int v = q.front();
//     q.pop();
//     vis[v] = 0;
//     int s = hg.start_pos[v];
//     int e = (v==(hg.nvertexs-1)?hg.nedges:hg.start_pos[v+1]);
//     for(int j=s; j<e; ++j){
//       int u = hg.adj_list[j];
//       int w = hg.edge_weights[j];
//       if(dis[u]>dis[v]+w){
//         dis[u]=dis[v]+w;
//         if(!vis[u]){
//           q.push(u);
//           vis[u]=1;
//         }
//       }
//     }
//   }
//   double me = mwtime();
//   LOG("CPU SSSP: %.3f ms\n",me-ms);
//   free(vis);
//   return dis;
// }


// void validation(int* dGPU, int* dCPU, int N){
//   bool flag=true;
//   //const float eps=1e-5;
//   for(int i=0;i<N;++i){
//     if(dGPU[i]-dCPU[i] != 0){
//       flag = false;
//       puts("failed");
//       std::cout << i << " " << dGPU[i] << " " << dCPU[i] << std::endl;
//       break;
//     }
//   }
//   if(flag) puts("passed");
// }

template<typename G, typename F>
double run_core(G g, F f){
  // step 1: initializing
  LOG(" -- Initializing\n");
  active_set_t as = build_active_set(g.nvertexs, conf);
//   as.init(ALL_ACTIVE);

  double s = mwtime();
  // step 2: Execute Algorithm
  // for(int level=0; ;level++){
  //   inspector.inspect(as, g, f, stats, fets, conf);
  //   if(as.finish(g,f,conf)) break;
  //   selector.select(stats, fets, conf);
  //   executor.filter(as, g, f, stats, fets, conf);
  //   executor.expand(as, g, f, stats, fets, conf);
  //   if(as.finish(g,f,conf)) break;
  //   g.update_level();
  // }

  for(int level=0; level < 500; level++) // 500 is manually set. 
  {
    while(true) // current level is done
    {
        inspector.inspect(as, g, f, stats, fets, conf);
        // if(as.finish(g,f,conf)) break;
        selector.select(stats, fets, conf);
        executor.filter(as, g, f, stats, fets, conf);
        executor.expand(as, g, f, stats, fets, conf);
        if(as.finish(g,f,conf)) break;
    }
    g.update_level();
  }

  double e = mwtime();

  return e-s;
}


int main(int argc, char* argv[]){
  parse_cmd(argc, argv, "KCORE");   

  // step 1 : set features
  fets.centric = VC;
  fets.pattern = ASSO;
  fets.fromall = false;
  fets.toall = true;

  // step 2 : init Graph & Algorithm
  edgelist_t<int> el;
  el.read_mtx(cmd_opt.path, cmd_opt.directed, cmd_opt.with_weight, cmd_opt.with_header);
  el.gen_weight(64);
  auto g = build_graph<VC,int>(el, fets);
  KCORE f;
  f.data.build(g.hg.nvertexs);

  // step 3 : choose root vertex
//   f.data.set_zero(0.0f);
  f.data.init_wa([g](int i){return g.hg.odegrees[i]; });
  f.data.init_ra([](int i){return -1; });
  init_conf(stats, fets, conf, g, f);

  // step 3 : execute Algorithm
  LOG(" -- Launching Kcore\n");
  double time = run_core(g.dg, f);
    
  // step 4 : validation
  f.data.sync_ra();
  f.data.sync_wa();
//   if(cmd_opt.validation){
//     validation(g.hg, f.data.h_ra);
//   }
  LOG("GPU Kcore time: %.3f ms\n", time);
  std::cout << time << std::endl;
  //std::cout << fets.nvertexs << " " 
            //<< fets.nedges << " "
            //<< fets.avg_deg << " "
            //<< fets.std_deg << " "
            //<< fets.range << " "
            //<< fets.GI << " "
            //<< fets.Her << " "
            //<< time << std::endl;

    // int max = -1;
    for(int i = 0; i < 40; ++i)
    {
        std::cout<< f.data.h_wa[i] << " ";
    }
    std::cout << std::endl;
    for(int i = 0; i < 40; ++i)
    {
        std::cout<< f.data.h_ra[i] << " ";
    }
    std::cout << std::endl;

  return 0;
}


