//
//  main.c
//  LB_Simulation
//
//  Created by Elgiz Bağcılar on 03.07.20.
//  Copyright © 2020 Elgiz Bağcılar. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <OpenCL/opencl.h>
#include <string.h>
#include "lb_kernel.cl.h"

#define NUM_CELL_X 400
#define NUM_CELL_Y 160



struct __attribute__ ((aligned (64))) cell{
    cl_double ro;
    cl_double u_x;
    cl_double u_y;
    cl_double f[9];
    cl_double f_old[9];
    cl_bool isB;
};

void create_vtk(int file_number,struct cell* cell_grid, int num_cell_x, int num_cell_y){
    
    char *vtk_file = (char *)malloc(sizeof(char) * 50);
    char f_number[10];
    sprintf(f_number, "%d", file_number);
    
    strcat(vtk_file,"/Output/lbmSim");
    strcat(vtk_file, f_number);
    strcat(vtk_file, ".vtk");
    FILE *vtk_ptr;
    vtk_ptr = fopen(vtk_file, "w+");
    
    if(vtk_ptr == NULL){
        printf("Error opening file.");
        exit(1);
    }
    fprintf(vtk_ptr, "# vtk DataFile Version 4.0\n");
    fprintf(vtk_ptr, "hesp visualization file\n");
    fprintf(vtk_ptr, "ASCII\n");
    fprintf(vtk_ptr, "DATASET RECTILINEAR_GRID\n");
    fprintf(vtk_ptr, "DIMENSIONS %d %d 1\n",num_cell_x,num_cell_y);
    
    fprintf(vtk_ptr, "X_COORDINATES %d int\n",num_cell_x);
    for (int i = 0; i < num_cell_x; ++i)
        fprintf(vtk_ptr,"%d ",i);
    fprintf(vtk_ptr,"\n");
    
    fprintf(vtk_ptr, "Y_COORDINATES %d int\n",num_cell_y);
    for (int i = 0; i < num_cell_y; ++i)
        fprintf(vtk_ptr,"%d ",i);
    fprintf(vtk_ptr,"\n");
    
    fprintf(vtk_ptr, "Z_COORDINATES 1 int\n");
    fprintf(vtk_ptr, "0\n");
    
    fprintf(vtk_ptr, "POINT_DATA %d\n", (num_cell_x * num_cell_y));
    
    fprintf(vtk_ptr, "FIELD FieldData 1\n");
    fprintf(vtk_ptr, "u 1 %d float\n", (num_cell_x * num_cell_y));
    for (int j = 0; j < num_cell_y; ++j){
        for (int i = 0; i < num_cell_x; ++i)
            fprintf(vtk_ptr, "%f ", cell_grid[i+j*num_cell_x].u_x);
        fprintf(vtk_ptr,"\n");
    }
    
    fprintf(vtk_ptr, "FIELD FieldData 1\n");
    fprintf(vtk_ptr, "ro 1 %d float\n", (num_cell_x * num_cell_y));
    for (int j = 0; j < num_cell_y; ++j){
        for (int i = 0; i < num_cell_x; ++i)
            fprintf(vtk_ptr, "%f ", cell_grid[i+j*num_cell_x].ro);
        fprintf(vtk_ptr,"\n");
    }
        
    fclose(vtk_ptr);
}



int main(int argc, const char * argv[]) {
   
    char name[128];
    double ux_inlet = atof(argv[1]);
    int direction[18] = { -1, -1 , 0, -1, 1, -1, -1, 0, 0, 0, 1, 0, -1, 1, 0, 1, 1, 1};
    double t[9]=  {1.0/36.0, 1.0/9.0, 1.0/36.0, 1.0/9.0, 4.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0};
    
    dispatch_queue_t queue =
              gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    if (queue == NULL) {
       queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    }
    // For getting device info if needed
    cl_device_id gpu = gcl_get_device_id_with_dispatch_queue(queue);
    clGetDeviceInfo(gpu, CL_DEVICE_NAME, 128, name, NULL);
    
    struct cell* cell_grid = (struct cell*)malloc(sizeof(struct cell) * NUM_CELL_X * NUM_CELL_Y);
    //Initialize cell grid
    for (int i = 0; i < NUM_CELL_X * NUM_CELL_Y; ++i) {
        cell_grid[i].ro = 0.0;
        cell_grid[i].u_x = 0.0;
        cell_grid[i].u_y = 0.0;
        for ( int j = 0; j < 9; ++j) {
            cell_grid[i].f[j] = t[j];
            cell_grid[i].f_old[j] = t[j];
        }
        cell_grid[i].isB = false;
    }
    //initialize the wall cells
    for (int i = 0; i < NUM_CELL_X; ++i) {
        cell_grid[i].isB = true;
        cell_grid[(NUM_CELL_X * NUM_CELL_Y)-i-1].isB = true;
    }
    
    //initialize the inlet velocity
    for(int i=2; i < (NUM_CELL_Y-2); ++i){
        cell_grid[i * NUM_CELL_X].u_x = ux_inlet;
    }
    
    void* cell_in = gcl_malloc(sizeof(struct cell) * NUM_CELL_X * NUM_CELL_X, cell_grid, CL_MEM_READ_WRITE |                                                                                                CL_MEM_COPY_HOST_PTR);
                           // CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* dir_in = gcl_malloc(sizeof(cl_int) * 18, direction,
                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* t_in =  gcl_malloc(sizeof(cl_double) * 9, t,
                            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    
    
    
    dispatch_sync(queue, ^{
       size_t wgs;
       gcl_get_kernel_block_workgroup_info(lb_kernel, CL_KERNEL_WORK_GROUP_SIZE,
                                           sizeof(wgs), &wgs, NULL);
       gcl_get_kernel_block_workgroup_info(collide_kernel, CL_KERNEL_WORK_GROUP_SIZE,
                                           sizeof(wgs), &wgs, NULL);
       
       cl_ndrange range = {1, {0, 0, 0}, {NUM_CELL_X * NUM_CELL_Y, 0, 0}, {wgs, 0, 0}};
       cl_ndrange range2 = {1, {0, 0, 0}, {NUM_CELL_X * NUM_CELL_Y, 0, 0}, {wgs, 0, 0}};
       
       for ( int i = 0; i < 1000 ; ++i) {
            lb_kernel(&range, cell_in, (cl_int*)dir_in, NUM_CELL_X, NUM_CELL_Y);
            collide_kernel(&range2, cell_in, (cl_int*)dir_in, NUM_CELL_X, NUM_CELL_Y, (cl_double*)t_in, (cl_int)i);
            gcl_memcpy(cell_grid, cell_in, sizeof(struct cell) * NUM_CELL_X * NUM_CELL_Y);
            if( i % 10 == 0) {
                create_vtk(i, cell_grid, NUM_CELL_X, NUM_CELL_Y);
            }
        }
     
    });
    
    gcl_free(cell_in);
    gcl_free(dir_in);
    gcl_free(t_in);

    free(cell_grid);

    dispatch_release(queue);
    
    printf("Simulation is completed.\n");
    return 0;
}
