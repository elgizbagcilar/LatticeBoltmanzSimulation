// kernel functions
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef struct __attribute__ ((aligned (64))) _cell{
    double ro;
    double u_x;
    double u_y;
    double f[9];
    double f_old[9];
    bool isB;
}cell;


kernel void lb(global cell* cells, global const int* dir, private int num_cell_x, private int num_cell_y){
    
    int idx = get_global_id(0);

    if(idx < num_cell_x * num_cell_y){
        
        for(int i=0; i<9; ++i){
            
            if(i==4) continue;
            
            // calculate neigh index
            int neigh_i = idx + dir[2*i] + dir[2*i+1] * num_cell_x;
            
            // correct neigh index for boundary cells
            if (idx % num_cell_x == 0 && dir[2*i] == -1){
                continue;
            }
            else if (idx % num_cell_x == num_cell_x-1 && dir[2*i] == 1){
                continue;
            }
            
            if (idx / num_cell_x==0 && dir[2*i+1]==-1) {
                neigh_i += num_cell_x * num_cell_y;
            }
            else if (idx / num_cell_x==num_cell_y-1 && dir[2*i+1]== 1) {
                neigh_i -= num_cell_x * num_cell_y;
            }
            
            // if neigh is not a wall --> normal copy
            if (cells[neigh_i].isB == false) {
                cells[neigh_i].f[i] = cells[idx].f_old[i];
            }
            else if (cells[neigh_i].isB == true && cells[idx].isB == false) {
                // if neigh is a wall and current cell is not a wall --> reflective copy
                cells[neigh_i].f[8-i] = cells[idx].f_old[i];
            }
        }
    }
}


kernel void collide(global cell* cells, global const int* dir, private int num_cell_x, private int num_cell_y,                                                                             global double* t, private int timeStep) {
    int idx = get_global_id(0);
    
    if(idx < num_cell_x*num_cell_y && cells[idx].isB == false){
        float f_eq[9];
        if(idx % num_cell_x != 0 ){
            //ro calculation
            if (timeStep!=0) {
                cells[idx].ro=0.0;
            }
            for (int i = 0; i < 9; ++i) {
                cells[idx].ro += cells[idx].f[i];
            }
            //u calculation
            cells[idx].u_x = 0.0;
            cells[idx].u_y = 0.0;
            for ( int i = 0; i < 9; ++i) {
                cells[idx].u_x += cells[idx].f[i] * dir[2*i];
                cells[idx].u_y += cells[idx].f[i] * dir[2*i+1];
            }
            cells[idx].u_x = cells[idx].u_x / cells[idx].ro;
            cells[idx].u_y = cells[idx].u_y / cells[idx].ro;
        }
        else{
            cells[idx].ro = ( 1/(1-cells[idx].u_x) ) * ( cells[idx].f[1] + cells[idx].f[4] + cells[idx].f[7] +                                               2 * ( cells[idx].f[0] + cells[idx].f[3] + cells[idx].f[6]));            
            //f calculation
            cells[idx].f[5] = cells[idx].f[3] + 2.0/3.0 * cells[idx].ro * cells[idx].u_x;
            cells[idx].f[8] = cells[idx].f[0] - (0.5 * (cells[idx].f[7] - cells[idx].f[1])) +
                                                (0.5 *  cells[idx].ro * cells[idx].u_y ) +
                                                (1.0/6.0 * cells[idx].ro * cells[idx].u_x );
            cells[idx].f[2] = cells[idx].f[6] + (0.5 * (cells[idx].f[7] - cells[idx].f[1]) ) -                                                            (0.5 * cells[idx].ro * cells[idx].u_y) +
                                                (1.0/6.0 * cells[idx].ro * cells[idx].u_x);
        }
        
        //f collide calculation
        for(int i=0;i<9;++i){
            f_eq[i] = (t[i] * cells[idx].ro) * ( 1.0 +
                                                 3.0 * (dir[2*i]*cells[idx].u_x+dir[2*i+1]*cells[idx].u_y) +
                                                 9.0/2.0 * ( ( dir[2*i] * cells[idx].u_x + dir[2*i+1] * cells[idx].u_y) *         ( dir[2*i] * cells[idx].u_x + dir[2*i+1] * cells[idx].u_y)) -
                                                 3.0/2.0 * ( cells[idx].u_x * cells[idx].u_x + cells[idx].u_y *                   cells[idx].u_y) );
            cells[idx].f[i] = cells[idx].f[i] - 1.2 * ( cells[idx].f[i] - f_eq[i]);
            cells[idx].f_old[i] = cells[idx].f[i];
        }
        
    }
    
}
