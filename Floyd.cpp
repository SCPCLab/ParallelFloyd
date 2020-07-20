#include<iostream>
#include<ctime>
#include<cstdio>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
using namespace std;

#define INDEX(i,j,ldy)  ( (i) * (ldy) + (j) ) 
#define TILE_WIDTH 32
#define NUM_NODE 40000
#define BigPart 2500
#define SmallPart ( (NUM_NODE + TILE_WIDTH - 1) / TILE_WIDTH - BigPart )

void shortestPath_floyd_cpu(int num_node, int *path_node, float *shortLenTable) {
        for (int u = 0; u < num_node; ++u) {
              for (int v = 0; v < num_node; ++v) {
                      for (int w = 0; w < num_node; ++w) {
                              if (shortLenTable[INDEX(v, u, num_node)] + shortLenTable[INDEX(u, w, num_node)] < shortLenTable[INDEX(v, w, num_node)]) {
                                      shortLenTable[INDEX(v, w, num_node)] = shortLenTable[INDEX(v, u, num_node)] + shortLenTable[INDEX(u, w, num_node)];
                                      path_node[INDEX(v, w, num_node)] = path_node[INDEX(u, w, num_node)];
                              }
                      }
              }
        }
}

void creatMainBlock(int num_node, int* path_node, float *shortLenTable, int *d_mainPath, float *d_mainDist, int mainId) {
      int offset = mainId * TILE_WIDTH; // the offset in X or Y

      const int blockSize = TILE_WIDTH * TILE_WIDTH;
      int mainPath[blockSize];
      float mainDist[blockSize];

      for (int i = 0; i < TILE_WIDTH && offset + i < num_node; i++) {
            for (int j = 0; j < TILE_WIDTH && offset + j < num_node; j++) {
                  mainPath[i * TILE_WIDTH + j] = path_node[INDEX(offset + i, offset + j, num_node)];
                  mainDist[i * TILE_WIDTH + j] = shortLenTable[INDEX(offset + i, offset + j, num_node)];
            }
      }

      hipMemcpy(d_mainPath, mainPath, sizeof(int) * blockSize, hipMemcpyHostToDevice);
      hipMemcpy(d_mainDist, mainDist, sizeof(float) * blockSize, hipMemcpyHostToDevice);
}
void creatMainBlockFromGpu(int num_node, int* path_node_d, float *shortLenTable_d, int *d_mainPath, float *d_mainDist, int mainId,int k_num, int flag){
        int X_offset = mainId * TILE_WIDTH; // flag= 1: part 3 in gpu
        if(flag) X_offset = (mainId - BigPart)* TILE_WIDTH;
        int Y_offset = mainId * TILE_WIDTH;
        for (int i = 0; i < k_num; i++) {
              hipMemcpy(&d_mainDist[INDEX(i,0,TILE_WIDTH)], &shortLenTable_d[INDEX(X_offset+i,Y_offset,num_node)], sizeof(float) * k_num, hipMemcpyDeviceToDevice);
              hipMemcpy(&d_mainPath[INDEX(i,0,TILE_WIDTH)], &path_node_d[INDEX(X_offset+i,Y_offset,num_node)], sizeof(int) * k_num, hipMemcpyDeviceToDevice);
        }
}
 
void creatRowBlock(int num_node, int* path_node, float *shortLenTable, int *row_block_path, float *row_block_dist, int* row_block_path_d, float *row_block_dist_d, int main_Id) {
	int offset = main_Id * TILE_WIDTH; // the offset of main block in X or Y
	int n = (num_node + TILE_WIDTH - 1) / TILE_WIDTH;  // blockdim
	int blockSize = TILE_WIDTH * TILE_WIDTH;

	for (int k = 0; k < n; k++) {
          int start = k * blockSize; // the number m blocks start address    row_block[number block][x][y]  the priority: y > x > number block 
          for (int i = 0; i < TILE_WIDTH && offset + i < num_node/* bounce check */; i++) {
                for (int j = 0; j < TILE_WIDTH && k * TILE_WIDTH + j < num_node/* bounce check */; j++) {
                      row_block_path[start + i * TILE_WIDTH + j] = path_node[(offset + i) * num_node + k * TILE_WIDTH + j];
                      row_block_dist[start + i * TILE_WIDTH + j] = shortLenTable[(offset + i) * num_node + k * TILE_WIDTH + j];
                }
          }
	}
	hipMemcpy(row_block_path_d, row_block_path, sizeof(int) * n * blockSize, hipMemcpyHostToDevice);
	hipMemcpy(row_block_dist_d, row_block_dist, sizeof(float) * n * blockSize, hipMemcpyHostToDevice);
}
void creatRowBlockFromGpu(int num_node, int* path_node_d, float *shortLenTable_d,  int* row_block_path_d, float *row_block_dist_d, int main_Id,int k_num, int flag){
      int offset = main_Id * TILE_WIDTH; // flag= 1: part 3 in gpu
      if(flag) offset = (main_Id - BigPart)* TILE_WIDTH;
      int n = (num_node + TILE_WIDTH - 1) / TILE_WIDTH;  // blockdim
      int blockSize = TILE_WIDTH * TILE_WIDTH;
      int t = TILE_WIDTH;
      for (int k = 0; k < n; k++) {
            int start = k * blockSize; // the number m blocks start address    row_block[number block][x][y]  the priority: y > x > number block 
            if(k == n - 1) t = num_node - k * TILE_WIDTH;
            for(int i = 0; i < k_num; ++i){
                  hipMemcpy(&row_block_path_d[start+i * TILE_WIDTH], &path_node_d[INDEX(offset + i, k * TILE_WIDTH, num_node)], sizeof(int) * t, hipMemcpyDeviceToDevice);
                  hipMemcpy(&row_block_dist_d[start+i * TILE_WIDTH], &shortLenTable_d[INDEX(offset + i, k * TILE_WIDTH, num_node)], sizeof(float) * t, hipMemcpyDeviceToDevice);
           	}
      }
}

void creatColBlockFromGpuAndCpu(int num_node, int* path_node, float* shortLenTable, int* path_node_d,float* shortLenTable_d, int* col_block_path_d, float* col_block_dist_d, int main_Id,int k_num,int flag){
        int offset = main_Id * TILE_WIDTH; // the offset of main block in X or Y
        int n = (num_node + TILE_WIDTH - 1) / TILE_WIDTH;  // blockdim
        int blockSize = TILE_WIDTH * TILE_WIDTH;
        int start;
        for(int k = 0; k < SmallPart; k++){
                if(flag){ // part 1 in host
                        start = k * blockSize; // the number m blocks start address    row_block[number block][x][y]  the priority: y > x > number block 
                        for(int i=0;i<TILE_WIDTH;++i){
                              hipMemcpy(&col_block_path_d[start+i*TILE_WIDTH], &path_node[INDEX(k*TILE_WIDTH+i,offset,num_node)], sizeof(int) * k_num, hipMemcpyHostToDevice);
                              hipMemcpy(&col_block_dist_d[start+i*TILE_WIDTH], &shortLenTable[INDEX(k*TILE_WIDTH+i,offset,num_node)], sizeof(float) *k_num,hipMemcpyHostToDevice);
                        }
                } else{ // part 1 in device
                        start = k * blockSize; // the number m blocks start address    row_block[number block][x][y]  the priority: y > x > number block 
                        for(int i=0;i<TILE_WIDTH;++i){
                                hipMemcpy(&col_block_path_d[start+i*TILE_WIDTH], &path_node_d[INDEX(k*TILE_WIDTH+i,offset,num_node)], sizeof(int) * k_num, hipMemcpyDeviceToDevice);
                                hipMemcpy(&col_block_dist_d[start+i*TILE_WIDTH], &shortLenTable_d[INDEX(k*TILE_WIDTH+i,offset,num_node)], sizeof(float) *k_num, hipMemcpyDeviceToDevice);
                        }
                }
        }
        for (int k = SmallPart; k < BigPart; k++) {
                start = k * blockSize; // the number m blocks start address    row_block[number block][x][y]  the priority: y > x > number block 
                for(int i=0;i<TILE_WIDTH;++i){
                        hipMemcpy(&col_block_path_d[start+i*TILE_WIDTH], &path_node_d[INDEX(k*TILE_WIDTH+i,offset,num_node)], sizeof(int) * k_num, hipMemcpyDeviceToDevice);
                        hipMemcpy(&col_block_dist_d[start+i*TILE_WIDTH], &shortLenTable_d[INDEX(k*TILE_WIDTH+i,offset,num_node)], sizeof(float) *k_num, hipMemcpyDeviceToDevice);
                }
        }
        int height = TILE_WIDTH;
        for (int k = BigPart; k < n; k++) {
        		if(k == n - 1) height = num_node - k * TILE_WIDTH;
                if(flag){ // part 3 in device
                        start = k * blockSize; // the number m blocks start address    row_block[number block][x][y]  the priority: y > x > number block 
                        for(int i = 0; i < height; ++i){
                                hipMemcpy(&col_block_path_d[start+i*TILE_WIDTH], &path_node_d[INDEX((k - BigPart) * TILE_WIDTH + i, offset, num_node)], sizeof(int) * k_num, hipMemcpyDeviceToDevice);
                                hipMemcpy(&col_block_dist_d[start+i*TILE_WIDTH], &shortLenTable_d[INDEX((k - BigPart) * TILE_WIDTH + i, offset, num_node)], sizeof(float) *k_num, hipMemcpyDeviceToDevice);
                        } 
                } else{ // part 3 in host
                        start = k * blockSize; // the number m blocks start address    row_block[number block][x][y]  the priority: y > x > number block 
                        for(int i = 0; i < height; ++i){
                                hipMemcpy(&col_block_path_d[start+i*TILE_WIDTH], &path_node[INDEX(k*TILE_WIDTH+i,offset,num_node)], sizeof(int) * k_num, hipMemcpyHostToDevice);
                                hipMemcpy(&col_block_dist_d[start+i*TILE_WIDTH], &shortLenTable[INDEX(k*TILE_WIDTH+i,offset,num_node)], sizeof(float) *k_num, hipMemcpyHostToDevice);
                        }
                }
        }
}
__global__ void initPath(int num_node, int* path_node_d) {
		int tx = threadIdx.x; int ty = threadIdx.y;
        int bx = blockIdx.x;  int by = blockIdx.y;

        if(bx * TILE_WIDTH + tx < num_node && by * TILE_WIDTH + ty < num_node) {
        		path_node_d[(bx * TILE_WIDTH + tx) * num_node + by * TILE_WIDTH + ty] = by * TILE_WIDTH + ty;
      	}
}
__global__ void step_One(int num_node, int* d_mainPath, float *d_mainDist, int k_num) {
		int tx = threadIdx.x; int ty = threadIdx.y;
        const int block_size = TILE_WIDTH * TILE_WIDTH;
        __shared__  int s_path[block_size];
        __shared__  float s_shortlen[block_size];

        int local_pos = tx * TILE_WIDTH + ty;

        s_path[local_pos] = d_mainPath[local_pos];
        s_shortlen[local_pos] = d_mainDist[local_pos];
        //__syncthreads();
        for (int k = 0; k < k_num; k++) {
        		if (s_shortlen[tx * TILE_WIDTH + k] + s_shortlen[k * TILE_WIDTH + ty] < s_shortlen[local_pos]) {
                		s_shortlen[local_pos] = s_shortlen[tx * TILE_WIDTH + k] + s_shortlen[k * TILE_WIDTH + ty];
                        s_path[local_pos] = s_path[k * TILE_WIDTH + ty];//
                }
               //	__syncthreads();
        }
        d_mainPath[local_pos] = s_path[local_pos];
        d_mainDist[local_pos] = s_shortlen[local_pos];
		//__syncthreads();
}

__global__ void step_Two(int num_node, int* row_block_path_d, float *row_block_dist_d, int* col_block_path_d, float *col_block_dist_d, int *d_mainPath, float *d_mainDist, int mainId, int k_num) {
		int tx = threadIdx.x; int ty = threadIdx.y;

        const int block_size = TILE_WIDTH * TILE_WIDTH;
        __shared__  int main_path[block_size];
        __shared__  float main_dist[block_size];

        __shared__  int s_path[block_size];
        __shared__  float s_dist[block_size];

        int bx = blockIdx.x; int by = blockIdx.y;
        int local_pos = tx * TILE_WIDTH + ty;
        
        main_path[local_pos] = d_mainPath[local_pos];
        main_dist[local_pos] = d_mainDist[local_pos]; // load main block
       
        // blockdim in Y = n - 1 (not include main block)

        // there is no bounce check ,because it in process of buffer to host memery
        if (bx == 0) { //caculate row block 
        		if (mainId * TILE_WIDTH + tx < num_node && by * TILE_WIDTH + ty < num_node) {
                		s_path[local_pos] = row_block_path_d[by * block_size + local_pos];
                        s_dist[local_pos] = row_block_dist_d[by * block_size + local_pos];
                        //__syncthreads();
                        for (int k = 0; k < k_num; k++) {
                                if (main_dist[tx * TILE_WIDTH + k] + s_dist[k * TILE_WIDTH + ty] < s_dist[local_pos]) {
                                        s_dist[local_pos] = main_dist[tx * TILE_WIDTH + k] + s_dist[k * TILE_WIDTH + ty];
                                        s_path[local_pos] = s_path[k * TILE_WIDTH + ty];
                                }
                                //__syncthreads();
                        }
                        row_block_path_d[by * block_size + local_pos] = s_path[local_pos];
                        row_block_dist_d[by * block_size + local_pos] = s_dist[local_pos];
            	}
        }
        else { // //caculate col block 
        		if (mainId * TILE_WIDTH + ty < num_node && by * TILE_WIDTH + tx < num_node) {
                		s_path[local_pos] = col_block_path_d[by * block_size + local_pos];
                        s_dist[local_pos] = col_block_dist_d[by * block_size + local_pos];
                       // __syncthreads();
                        for (int k = 0; k < k_num; k++) {
                        		if (s_dist[tx * TILE_WIDTH + k] + main_dist[k * TILE_WIDTH + ty] < s_dist[local_pos]) {
                                		s_dist[local_pos] = s_dist[tx * TILE_WIDTH + k] + main_dist[k * TILE_WIDTH + ty];
                                        s_path[local_pos] = main_path[k * TILE_WIDTH + ty];
                                }
                                //__syncthreads();
                        }
                        col_block_path_d[by * block_size + local_pos] = s_path[local_pos];
                        col_block_dist_d[by * block_size + local_pos] = s_dist[local_pos];
            	}
        }
       // __syncthreads();
}

__global__ void step_Three(int num_node, int *row_block_path_d, float* row_block_dist_d, int *col_block_path_d,  float *col_block_dist_d, int *path_node_d, float *shortLenTable_d, int main_Id, int k_num, int flag) {
		const int block_size = TILE_WIDTH * TILE_WIDTH;
        int bx = blockIdx.x; int by = blockIdx.y;
        int gx = bx;
		if(flag && gx < SmallPart) gx += BigPart;
    
        int row_offset = by * block_size; //row block offset 
        int col_offset =  gx* block_size;

        int tx = threadIdx.x; int ty = threadIdx.y;
        int local_pos = tx * TILE_WIDTH + ty;
        int pos = (bx * TILE_WIDTH + tx) * num_node + by * TILE_WIDTH + ty;

        __shared__ int myPath[block_size]; // store the block to calculate, the same row data in col_block_dist_d, the same col data in row_block_dist_d
        __shared__ float myDist[block_size];

        __shared__ int rowPath[block_size];
        __shared__ float rowDist[block_size];

        __shared__ int colPath[block_size];
        __shared__ float colDist[block_size];

        rowPath[local_pos] = row_block_path_d[row_offset + local_pos];
        rowDist[local_pos] = row_block_dist_d[row_offset + local_pos];

        colPath[local_pos] = col_block_path_d[col_offset + local_pos];
        colDist[local_pos] = col_block_dist_d[col_offset + local_pos];

        if (gx  * TILE_WIDTH + tx < num_node && by * TILE_WIDTH + ty < num_node /*bounce check*/) {
        		myPath[local_pos] = path_node_d[pos];
                myDist[local_pos] = shortLenTable_d[pos];
               // __syncthreads();

                for (int k = 0; k < k_num; k++) { // k_num: current main blocks width
                		if (colDist[tx * TILE_WIDTH + k] + rowDist[k * TILE_WIDTH + ty] < myDist[local_pos]) {
                        		myDist[local_pos] = colDist[tx * TILE_WIDTH + k] + rowDist[k * TILE_WIDTH + ty];
                            	myPath[local_pos] = rowPath[k * TILE_WIDTH + ty];
                        }
                   		 //__syncthreads();
                }
                path_node_d[pos] = myPath[local_pos];
                shortLenTable_d[pos] = myDist[local_pos];
        }
       // __syncthreads();
}
__global__ void step_OneNoDivide(int num_node, int* path_node_d, float *shortLenTable_d, int main_Id, int k_num) {
	int offset = TILE_WIDTH * main_Id; //主模块的水平或者垂直偏移
	int tx = threadIdx.x; int ty = threadIdx.y;
	int gx = offset + tx; int gy = offset + ty;
	
	int pos = (offset + tx) * num_node + offset  + ty;
	int locl_pos = tx * TILE_WIDTH + ty;
	
	const int block_size = TILE_WIDTH * TILE_WIDTH;
	__shared__  int myPath[block_size];
	__shared__  float myDist[block_size];

	if (gx < num_node && gy < num_node) {

		myPath[locl_pos] = path_node_d[pos];
		myDist[locl_pos] = shortLenTable_d[pos];
		
		__syncthreads();  //等待数据加载到共享内存 
		for (int k = 0; k < k_num; k++) {  //迭代次数 k_um
			if (myDist[tx * TILE_WIDTH + k] + myDist[k * TILE_WIDTH + ty] < myDist[locl_pos]) {
				myDist[locl_pos] = myDist[tx * TILE_WIDTH + k] + myDist[k * TILE_WIDTH + ty];
				myPath[locl_pos] = myPath[k * TILE_WIDTH + ty];//
			}
			__syncthreads();  //等待数据写回 
		}
		path_node_d[pos] = myPath[locl_pos];
		shortLenTable_d[pos] = myDist[locl_pos];
		
		__syncthreads();  //等待数据写回 
	}
}

__global__ void step_TwoNoDivide (int num_node, int* path_node_d, float *shortLenTable_d, int main_Id, int k_num) {
	int offset = TILE_WIDTH * main_Id; //主模块的水平或者垂直偏移

	int tx = threadIdx.x; int ty = threadIdx.y;
	int local_pos = tx * TILE_WIDTH + ty;

	const int block_size = TILE_WIDTH * TILE_WIDTH;
	__shared__  float mainDist[block_size]; //主模块
	__shared__  int mainPath[block_size];

	__shared__  int myPath[block_size];
	__shared__  float myDist[block_size]; //当前模块

	
	int bx = blockIdx.x; int by = blockIdx.y;
	
	if (tx < k_num && ty < k_num) { //加载主模块
		mainDist[local_pos]  = shortLenTable_d[(offset + tx) * num_node + offset + ty];
		mainPath[local_pos] = path_node_d[(offset + tx) * num_node + offset + ty];
	}

	if (bx == 0) { //计算与同行的块
		int pos = (offset + tx) * num_node + by * TILE_WIDTH + ty; //对应全局位置
		int gx = offset + tx; int gy = by * TILE_WIDTH + ty;
		if (gx < num_node && gy < num_node) { //边界检测
						
			myPath[local_pos] = path_node_d[pos];
			myDist[local_pos] = shortLenTable_d[pos];
			__syncthreads();  //等待数据加载到共享内存 
			
			for (int k = 0; k < k_num; k++) {
				if (mainDist[tx * TILE_WIDTH + k] + myDist[k * TILE_WIDTH + ty] < myDist[local_pos]) {
					myDist[local_pos] = mainDist[tx * TILE_WIDTH + k] + myDist[k * TILE_WIDTH + ty];
					myPath[local_pos] = myPath[k * TILE_WIDTH + ty];
				}
				__syncthreads();  //等待数据写回 
			}
			path_node_d[pos] = myPath[local_pos];
			shortLenTable_d[pos] = myDist[local_pos];
		}
	}else { // 计算与主模块同列的块
		int pos = (by * TILE_WIDTH + tx) * num_node + offset + ty; //对应全局位置
		int gx = by * TILE_WIDTH + tx; int gy = offset + ty;

		if (gx < num_node && gy < num_node) { // 边界检测

			myPath[local_pos] = path_node_d[pos];
			myDist[local_pos] = shortLenTable_d[pos];
			__syncthreads();  //等待数据加载到共享内存

			for (int k = 0; k < k_num; k++) {
				if (myDist[tx * TILE_WIDTH + k] + mainDist[k * TILE_WIDTH + ty] < myDist[local_pos]) {
					myDist[local_pos] = myDist[tx * TILE_WIDTH + k] + mainDist[k * TILE_WIDTH + ty];
					myPath[local_pos] = mainPath[k * TILE_WIDTH + ty];
				}
				__syncthreads();  //等待数据写回 
			}
			path_node_d[pos] = myPath[local_pos];
			shortLenTable_d[pos] = myDist[local_pos];
		}
	}
	__syncthreads(); //等待数据写回
}
__global__ void step_ThreeNoDivide(int num_node, int* path_node_d, float *shortLenTable_d, int main_Id, int k_num) {

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	int gx = bx * TILE_WIDTH + tx;  int gy = by * TILE_WIDTH + ty;
	int offset = TILE_WIDTH * main_Id; //主模块的偏移量

	int local_pos = tx * TILE_WIDTH + ty;

	const int block_size = TILE_WIDTH * TILE_WIDTH;
	__shared__ float colDist[block_size]; //同行参照

	if (bx * TILE_WIDTH + tx < num_node && offset + ty < num_node) {  // 边界检测
			colDist[local_pos] = shortLenTable_d[(bx * TILE_WIDTH + tx) * num_node + offset + ty];
	}
	
	__shared__ float rowDist[block_size]; //同列参照
	__shared__ int rowPath[block_size];

	if (offset + tx < num_node && by * TILE_WIDTH + ty < num_node) { // 边界检测
		rowDist[local_pos] = shortLenTable_d[(offset + tx) * num_node + by * TILE_WIDTH + ty];
		rowPath[local_pos] = path_node_d[(offset + tx) * num_node + by * TILE_WIDTH + ty];
	}
	
	if (gx < num_node && gy < num_node) {
		
		__shared__ float myDist[block_size]; //求解对象
		__shared__ int myPath[block_size];

		int pos = (bx * TILE_WIDTH + tx) * num_node + by * TILE_WIDTH + ty; // 线程对应global位置

		myDist[local_pos] = shortLenTable_d[pos];
		myPath[local_pos] = path_node_d[pos];

		__syncthreads(); //等待数据加载

		
		for (int k = 0; k < k_num; k++) {
			if (colDist[tx * TILE_WIDTH + k] + rowDist[k * TILE_WIDTH + ty] < myDist[local_pos]) {
				myDist[local_pos] = colDist[tx * TILE_WIDTH + k] + rowDist[k * TILE_WIDTH + ty];
				myPath[local_pos] = rowPath[k * TILE_WIDTH + ty];
			}
			__syncthreads(); //等待数据写回
		}
		shortLenTable_d[pos] = myDist[local_pos];
		path_node_d[pos] = myPath[local_pos];
	}
	__syncthreads(); //等待数据写回
}
void shortestPath_floyd_NoDivide(int num_node,  float *arc, int *path_node, float *shortLenTable){
		int *path_node_d; float *shortLenTable_d;    
        hipMalloc(&path_node_d, sizeof(int)*num_node*num_node);  hipMalloc(&shortLenTable_d, sizeof(float)*num_node*num_node);	
       
       	int n = (num_node + TILE_WIDTH - 1) / TILE_WIDTH;
		dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGridOne(1, 1);
        dim3 dimGridTwo(2, n);
        dim3 dimGridThree(n, n);
        hipMemcpy(shortLenTable_d, arc, sizeof(float) * num_node * num_node, hipMemcpyHostToDevice);
     	
        hipLaunchKernelGGL(initPath,dimGridThree,dimBlock,0,0,num_node, path_node_d);
        hipDeviceSynchronize();
        
        int k_num = TILE_WIDTH;
        for (int mainId = 0; mainId < n; mainId++) { 
				if (mainId == n - 1 && num_node % TILE_WIDTH != 0) k_num = num_node % TILE_WIDTH;
        		hipLaunchKernelGGL(step_OneNoDivide,dimGridOne,dimBlock,0,0,num_node, path_node_d, shortLenTable_d, mainId, k_num);
                hipDeviceSynchronize();

                hipLaunchKernelGGL(step_TwoNoDivide,dimGridTwo,dimBlock,0,0,num_node, path_node_d, shortLenTable_d,  mainId, k_num);
                hipDeviceSynchronize();

                hipLaunchKernelGGL(step_ThreeNoDivide,dimGridThree,dimBlock,0,0,num_node, path_node_d, shortLenTable_d, mainId, k_num);
                hipDeviceSynchronize();
        }        
        hipMemcpy(path_node, path_node_d, sizeof(int)*num_node*num_node, hipMemcpyDeviceToHost);
		hipMemcpy(shortLenTable, shortLenTable_d, sizeof(float)*num_node*num_node, hipMemcpyDeviceToHost);
        hipFree(path_node_d);
       	hipFree(shortLenTable_d);
}
void shortestPath_floyd(int num_node,  float *arc, int *path_node, float *shortLenTable){
        
        if(num_node <= BigPart * TILE_WIDTH){
        		shortestPath_floyd_NoDivide(num_node, arc, path_node, shortLenTable);
        } else{
                int *d_mainPath;  float *d_mainDist;       
                hipMalloc(&d_mainPath, sizeof(int) * TILE_WIDTH * TILE_WIDTH);  hipMalloc(&d_mainDist, sizeof(float) * TILE_WIDTH * TILE_WIDTH);

                int *path_node_d;  float *shortLenTable_d;

                hipMalloc(&path_node_d, sizeof(int) * BigPart * TILE_WIDTH * num_node);    hipMalloc(&shortLenTable_d, sizeof(float) * BigPart * TILE_WIDTH * num_node);

                int n = (num_node + TILE_WIDTH - 1) / TILE_WIDTH;

                //the col row block buffer in host 
                int size = n * TILE_WIDTH * TILE_WIDTH;

                int *row_block_path;  float *row_block_dist;
                int *col_block_path;   float *col_block_dist;

                row_block_path = (int*)malloc(sizeof(int) * size);    row_block_dist = (float*)malloc(sizeof(float) * size);
                col_block_path = (int*)malloc(sizeof(int) * size);  col_block_dist = (float*)malloc(sizeof(float) * size);


                int *row_block_path_d;  float *row_block_dist_d;
                int *col_block_path_d;    float *col_block_dist_d;

                hipMalloc(&row_block_path_d, sizeof(int) * size);     hipMalloc(&row_block_dist_d, sizeof(float) * size);
                hipMalloc(&col_block_path_d, sizeof(int) * size);   hipMalloc(&col_block_dist_d, sizeof(float) * size);

                int k_num = TILE_WIDTH;

                dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
                dim3 dimGridOne(1, 1);
                dim3 dimGridTwo(2, n);
                dim3 dimGridThreeBig(BigPart, n);
                dim3 dimGridThreeSmall(SmallPart, n);

                int topPartSize = SmallPart * TILE_WIDTH * num_node;
                int midPartSize = (BigPart - SmallPart) * TILE_WIDTH * num_node; 
                int bottomPartSize = (num_node - BigPart * TILE_WIDTH) * num_node;

                memcpy(&(shortLenTable[BigPart * TILE_WIDTH * num_node]), &(arc[BigPart * TILE_WIDTH * num_node]), sizeof(float) * bottomPartSize);
                hipMemcpy(shortLenTable_d, arc, sizeof(float) * BigPart * TILE_WIDTH * num_node, hipMemcpyHostToDevice);

                hipLaunchKernelGGL(initPath,dimGridThreeBig,dimBlock,0,0,num_node, path_node_d);
                hipDeviceSynchronize();
                hipMemcpy(&path_node[BigPart * TILE_WIDTH * num_node], path_node_d, sizeof(int) * bottomPartSize, hipMemcpyDeviceToHost); 

				
                for (int mainId = 0; mainId < n; mainId++) { 
                        if (mainId == n - 1 && num_node % TILE_WIDTH != 0) k_num = num_node % TILE_WIDTH;

                        //step_one
                      if(mainId < SmallPart){
                                if(mainId % 2 ){ // 1 in host , 3 in device
                                        creatMainBlock(num_node, path_node, shortLenTable, d_mainPath, d_mainDist, mainId);
                                } else{
                                        creatMainBlockFromGpu(num_node, path_node_d, shortLenTable_d, d_mainPath, d_mainDist, mainId,k_num, 0);
                                }
                        } else if (mainId < BigPart){
                                creatMainBlockFromGpu(num_node, path_node_d, shortLenTable_d, d_mainPath, d_mainDist, mainId,k_num, 0);
                        } else{
                                if(mainId % 2 ){
                                        creatMainBlockFromGpu(num_node, path_node_d, shortLenTable_d, d_mainPath, d_mainDist, mainId,k_num, 1);
                                } else{
                                        creatMainBlock(num_node, path_node, shortLenTable, d_mainPath, d_mainDist, mainId);
                                }
                        }

                        hipLaunchKernelGGL(step_One,dimGridOne,dimBlock,0,0,num_node, d_mainPath, d_mainDist, k_num);
                        hipDeviceSynchronize();
					
                        //step two

                        creatColBlockFromGpuAndCpu(num_node, path_node, shortLenTable,path_node_d,shortLenTable_d, col_block_path_d, col_block_dist_d, mainId,k_num, mainId % 2 );	
                        if(mainId < SmallPart){ 
                                if(mainId % 2 ){ // part 1 in host
                                        creatRowBlock(num_node, path_node, shortLenTable, row_block_path, row_block_dist, row_block_path_d, row_block_dist_d, mainId);
                                } else{
                                        creatRowBlockFromGpu(num_node, path_node_d, shortLenTable_d, row_block_path_d, row_block_dist_d, mainId,k_num, 0);
                                }
                        } else if(mainId < BigPart){
                                creatRowBlockFromGpu(num_node, path_node_d, shortLenTable_d, row_block_path_d, row_block_dist_d, mainId,k_num, 0);
                        } else{
                                if(mainId % 2 ){ // part 3 in device
                                        creatRowBlockFromGpu(num_node, path_node_d, shortLenTable_d, row_block_path_d, row_block_dist_d, mainId,k_num, 1);
                                } else{
                                        creatRowBlock(num_node, path_node, shortLenTable, row_block_path, row_block_dist, row_block_path_d, row_block_dist_d, mainId);
                                }
                        }


                        hipLaunchKernelGGL(step_Two,dimGridTwo,dimBlock,0,0,num_node, row_block_path_d, row_block_dist_d, col_block_path_d, col_block_dist_d, d_mainPath, d_mainDist, mainId, k_num);
                        hipDeviceSynchronize();
                        
                     
                        
                        //step Three

                        hipLaunchKernelGGL(step_Three,dimGridThreeBig,dimBlock,0,0,num_node, row_block_path_d,  row_block_dist_d, col_block_path_d, col_block_dist_d, path_node_d, shortLenTable_d, mainId, k_num, mainId % 2);
                        hipDeviceSynchronize();

                        if(mainId % 2 ){ //  3: gpu to cpu  1: cpu to gpu 
                                //copy back child block3
                                hipMemcpy(&path_node[BigPart* TILE_WIDTH * num_node], path_node_d, sizeof(int) * bottomPartSize, hipMemcpyDeviceToHost);
                                hipMemcpy(&shortLenTable[BigPart* TILE_WIDTH * num_node], shortLenTable_d, sizeof(float) * bottomPartSize, hipMemcpyDeviceToHost);

                                //copy child block1
                                hipMemcpy(path_node_d, path_node, sizeof(int) * topPartSize,hipMemcpyHostToDevice);
                                hipMemcpy(shortLenTable_d, shortLenTable, sizeof(float) * topPartSize,hipMemcpyHostToDevice);
                        } else{ // 1: gpu to cpu   3: cpu to gpu
                                //copy back child block1
                              hipMemcpy(path_node, path_node_d, sizeof(int) * topPartSize, hipMemcpyDeviceToHost);
                              hipMemcpy(shortLenTable, shortLenTable_d, sizeof(float) * topPartSize, hipMemcpyDeviceToHost);

                              //copy child block3
                              hipMemcpy(path_node_d, &path_node[BigPart* TILE_WIDTH * num_node], sizeof(int) * bottomPartSize, hipMemcpyHostToDevice);
                              hipMemcpy(shortLenTable_d, &shortLenTable[BigPart* TILE_WIDTH * num_node], sizeof(float) * bottomPartSize, hipMemcpyHostToDevice);
                        }

                        hipLaunchKernelGGL(step_Three,dimGridThreeSmall,dimBlock,0,0,num_node, row_block_path_d, row_block_dist_d, col_block_path_d, col_block_dist_d, path_node_d, shortLenTable_d, mainId, k_num, !(mainId % 2));
                        hipDeviceSynchronize();
						
                     
                        if(mainId == n - 1){
                                // copy out 2
                                hipMemcpy(&path_node[topPartSize], &path_node_d[topPartSize], sizeof(int) * midPartSize, hipMemcpyDeviceToHost);
                                hipMemcpy(&shortLenTable[topPartSize], &shortLenTable_d[topPartSize], sizeof(float) * midPartSize, hipMemcpyDeviceToHost);
                                if(mainId % 2 ){ // copy out 1
                                        hipMemcpy(path_node, path_node_d, sizeof(int) * topPartSize, hipMemcpyDeviceToHost);
                                        hipMemcpy(shortLenTable, shortLenTable_d, sizeof(float) * topPartSize, hipMemcpyDeviceToHost);	
                                } else{ // copy out 3
                                        hipMemcpy(&path_node[BigPart* TILE_WIDTH * num_node], path_node_d, sizeof(int) * bottomPartSize, hipMemcpyDeviceToHost);
                                        hipMemcpy(&shortLenTable[BigPart* TILE_WIDTH * num_node], shortLenTable_d, sizeof(float) * bottomPartSize, hipMemcpyDeviceToHost);
                                }
                        }	
                } 
                free(row_block_path);
                free(row_block_dist);
                free(col_block_path);
                free(col_block_dist);

                hipFree(path_node_d);
                hipFree(shortLenTable_d);
                hipFree(d_mainPath);
                hipFree(d_mainDist);
                hipFree(row_block_path_d);
                hipFree(row_block_dist_d);
                hipFree(col_block_path_d);
                hipFree(col_block_dist_d);
		}
}

void printResult(int num_node, int *path_node, float *shortLenTable) {
		for (int v = 0; v < num_node; ++v) {
        		for (int w = 0; w < num_node; ++w) {
                		if (v == w) continue;
                        printf("V%d - V%d dist:%lf \n", v, w, shortLenTable[v * num_node + w]);
                        int k = path_node[v * num_node + w];
                        printf("path: %d", v);
                        while (k != w) {
                        		printf(" -> %d", k);
                            	k = path_node[k * num_node + w];
                        }
                        printf(" -> %d\n\n", w);
        		}
		}
}

int check(int num_node, int *path_node, float *shortLenTable, int *gpu_path_node, float *gpu_shortLenTable) {
		int flag = 1;                                                      // mode: 0 check mainblock 1: check row col block 2: check all block
        for (int v = 0; v < num_node; ++v) {
              	for (int w = 0; w < num_node; ++w) {
                        if (!(fabs(gpu_shortLenTable[INDEX(v, w, num_node)] - shortLenTable[INDEX(v, w, num_node)]) < 1e-6)) {
                        		printf("The dist  is error\n");
                                printf("v: %d w: %d gpu:%f cpu:%f\n", v, w, gpu_shortLenTable[INDEX(v, w, num_node)], shortLenTable[INDEX(v, w, num_node)]);
                                flag = 0; return flag;
                        }
                        if (gpu_path_node[INDEX(v, w, num_node)] != path_node[INDEX(v, w, num_node)]) {
                        		printf("The path  is error\n");
                                printf("v: %d w: %d gpu: %d cpu: %d \n", v, w, gpu_path_node[INDEX(v, w, num_node)], path_node[INDEX(v, w, num_node)]);
                                flag = 0;  return flag;
                        }
            	}
        }
        return flag;
}

int main() {
		
		int num_node = NUM_NODE;
        float *shortLenTable;
        float *arc;
        int* path_node;
        
        shortLenTable = (float*)malloc(sizeof(float) * num_node * num_node);
        arc = (float*)malloc(sizeof(float) * num_node * num_node);
        path_node = (int*)malloc(sizeof(int) * num_node * num_node);
        
       	/*float *cpu_shortLenTable;
 		int   *cpu_path_node;
        cpu_shortLenTable = (float*)malloc(sizeof(float)	* num_node * num_node);
        cpu_path_node = (int*)malloc(sizeof(int)		* num_node * num_node);
        */
       	srand(time(NULL));
        for (int i = 0; i < num_node; i++) {
        		for (int j = 0; j < num_node; j++) { 
                		//cpu_path_node[INDEX(i, j, num_node)] = j; 
                		if (i == j) {
                              	arc[INDEX(i, j, num_node)] = 0;
                                //cpu_shortLenTable[INDEX(i, j, num_node)] = 0;
                                continue;
                        }
                        int randnum = rand() % 100;
                        if (randnum % 5 == 0) {
                        		arc[INDEX(i, j, num_node)] = (float)randnum / 10 + 1;
                        }else {
                            	arc[INDEX(i, j, num_node)] = 100;
                        }
                        //cpu_shortLenTable[INDEX(i, j, num_node)] = arc[INDEX(i, j, num_node)];
                 } 
        } 
        
        printf("           the number of node: %d \n", num_node);

		clock_t start, end;
        printf("*****************GPU is runing***************\n\n");
        start = clock();

		shortestPath_floyd(num_node, arc, path_node, shortLenTable);
        end = clock();
        printf("GPU time: %lf S\n", (double)(end - start) / CLOCKS_PER_SEC);
       
       /* 
		FILE *fp = fopen("myRecord.txt", "a");
        if(fp == NULL) exit(0);
        else{
        		fprintf(fp, "The number of node is %d, the GPU run time is %lf S\n", num_node,  (double)(end - start) / CLOCKS_PER_SEC);
        		fclose(fp);
        }
       */
       
       /* printf("*****************CPU is runing***************\n\n");
        start = clock();

        shortestPath_floyd_cpu(num_node, cpu_path_node, cpu_shortLenTable);

        end = clock();
        printf("CPU time: %lf S\n", (double)(end - start) / CLOCKS_PER_SEC);
        
      	if (!check(num_node, cpu_path_node, cpu_shortLenTable, path_node, shortLenTable)) exit(0);
        printf("the anwser is right\n");  
		*/
        free(arc);
        free(path_node);
        free(shortLenTable);
        
        return 0;
}
