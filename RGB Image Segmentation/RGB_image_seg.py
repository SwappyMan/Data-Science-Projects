
"""
Description: This module contains functions which allow a given user to find a suitable sparse cut
of a given image. 
"""
import numpy as np
import cv2
from scipy.sparse import csr_matrix

#Function which downsamples a given image to 100x100 pixels
def down_sampled_image(file_path):
    
    # adjust width and height to 100x100
    dim = (100,100)

    #Reading the image and downsizing
    open_cv_image_1 = cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
    open_cv_image = cv2.resize(src = open_cv_image_1, dsize = dim, interpolation = cv2.INTER_CUBIC )

    # Gaussian Blurring 
    gausBlur = cv2.GaussianBlur(open_cv_image,(5,5),0)
    
    return gausBlur

#Function which normalises the (r,g,b,x,y) co-ordinates
def normalise_gaus_blur_cordinates(gausBlur):
    
    rgbxy = np.zeros((100,100,5))
    gausBlur_norm = gausBlur/256
    
    #Normalising the (r,g,b,x,y) co-ordinates
    for i in range(0,len(gausBlur_norm),1):
        for j in range(0,len(gausBlur_norm[i]),1):
            i_norm = i/100
            j_norm = j/100
            for k in range(0,len(gausBlur_norm[i][j]),1):
                rgbxy[i][j][k] = gausBlur_norm[i][j][k]
            rgbxy[i][j][3] = j_norm
            if i >94:
                rgbxy[i][j][4] = i_norm - 1
            else:
                rgbxy[i][j][4] = i_norm
    
            if j >94:
                rgbxy[i][j][3] = j_norm - 1
            else:
                rgbxy[i][j][3] = j_norm
    return rgbxy

#Calculates the weights between pixels 
def weights(a,b):
    return np.exp(-4*((np.linalg.norm(b - a,axis=1))**2))

#Function which constructs the adjacency matrix
def construct_adjacency_matrix(normalise_cordinates): 
    
    #row = np.array([])
    #col = np.array([])
    #data = np.array([]) 

    master_col = []
    master_row = []
    master_data = []
    
    #Iterates through each pixel within the given picture through the (r,g,b,x,y) co-ordinate system
    for y in range(0,len(normalise_cordinates),1):
        for x in range(0,len(normalise_cordinates[y]),1):
            a = normalise_cordinates.take(range(y-5,y+6),mode='wrap', axis=0).take(range(x-5,x+6),mode='wrap',axis=1)[:,:,0:5] #clip clip
            c = a.reshape((-1, 5))
            d = np.unique(c,axis = 0)
            e = np.where(np.all(d == a[5][5], axis=1))
            f = np.delete(d, e, axis=0)
            
            X = np.repeat(a[5][5][None,:], f.shape[0], axis=0)
                
            po = weights(f,X)
            po.flatten()
            first = f[:,3:5]*100 
                
            first[first < 0] += 100
            row = (100*first[:,1:2]+first[:,0:1]).reshape((-1,))
            row_con = [int(round(x)) for x in row]
            master_row.append(row_con)
                
            centre_pixel = y*(100) + x
            col_1 = np.full(shape = f.shape[0], fill_value = centre_pixel, dtype=np.int)
            col = col_1.reshape((-1,))
            col_con = [int(round(x)) for x in col]
            master_col.append(col_con)
                    
            data = po.reshape((-1,))
            data_con = [x for x in data]
            master_data.append(data_con)
  
    row_final = [val for sublist in master_row for val in sublist]
    col_final = [val for sublist in master_col for val in sublist]
    data_final = [val for sublist in master_data for val in sublist]
    
    row_final_numpy = np.array(row_final)
    col_final_numpy = np.array(col_final)
    data_final_numpy = np.array(data_final)
    
    adjacency_matrix = csr_matrix((data_final_numpy, (row_final_numpy, col_final_numpy)), shape=(10000, 10000)).toarray()
    adjacency_matrix[adjacency_matrix < 0.9] = 0
    
    no_edges = np.count_nonzero(adjacency_matrix)

    return adjacency_matrix, no_edges, no_edges/10000

#Function which calculates the eignevector using the power method
def construct_heatmap_matrix(adjacency_matrix):
    
    m = np.asmatrix(adjacency_matrix)
    new_matrix = np.zeros((10000,10000))

    new_matrix = -1/np.sqrt(m.sum(axis=1)*m.sum(axis=0))

    new_matrix[adjacency_matrix==0] = 0
    np.fill_diagonal(new_matrix,1)

    fle = np.sqrt(m.sum(axis=0)).transpose()

    pop = 2*np.identity(10000)-new_matrix

    x = np.random.normal(loc = 0, scale = 1, size = (10000,1))
    er = np.dot(fle.transpose(),x)

    xe = x - np.multiply(er,fle)
    
    K = 175

    
    for i in range(1,K,1):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(pop, xe)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)
    
        # re normalize the vector
        xe = b_k1 / b_k1_norm
        
    heatmap_matrix = np.reshape(b_k1, (100, 100))

    return heatmap_matrix

#Function which finds a suitable sparse cut given the adjacency matrix and eigenvector
def construct_sparse_cut(adjacency_matrix,heatmap_matrix):
    
    flattened_heatmap = heatmap_matrix.flatten().tolist()
    flattened_heatmap_list = [val for sublist in flattened_heatmap for val in sublist]
    flattened_heatmap_np = np.array(flattened_heatmap_list).transpose()
    vertices = [i for i in range(0,len(flattened_heatmap_list),1)]
    
    adjacency_matrix_sum = adjacency_matrix.sum()
    adjacency_matrix_sum_diag = (adjacency_matrix.sum(axis = 1))**-0.5
    D = np.diag(adjacency_matrix_sum_diag)
    flattened = np.dot(D,flattened_heatmap_np)
    flattened_heatmap_list_sorted, vertices_sorted = zip(*sorted(zip(flattened, vertices)))
    
    t = 0
    n = len(vertices_sorted)
    s = []
    s_star = [vertices_sorted[0]]
    pointer = False
    w_cum_sum = 0
    w_cum_sum_st = 0
    vol_S = 0
    vol_S_st = 0
    blah = []
    blah_2 = []
    while t < n-1:
        s.append(vertices_sorted[t])

        adjacency_matrix_row = adjacency_matrix[vertices_sorted[t]].flatten().reshape((1,-1))
        adjacency_matrix_row_sum = adjacency_matrix_row.sum()
        res = adjacency_matrix_row_sum - np.delete(adjacency_matrix_row,s).sum()
        w_cum_sum = w_cum_sum + adjacency_matrix_row_sum - 2*res
        vol_S = vol_S + adjacency_matrix_row_sum
        vol_S_rem = adjacency_matrix_sum - vol_S
        phi_s = float(w_cum_sum/min(vol_S,vol_S_rem))
        blah.append(phi_s)
    
        if pointer == True:
            pass
        elif pointer == False:
            adjacency_matrix_row_st = adjacency_matrix[s_star[-1]].flatten().reshape((1,-1))
            adjacency_matrix_row_sum_st = adjacency_matrix_row_st.sum()
            res_st = adjacency_matrix_row_sum_st - np.delete(adjacency_matrix_row_st,s_star).sum()
            w_cum_sum_st = w_cum_sum_st + adjacency_matrix_row_sum_st - 2*res_st
            vol_S_st = vol_S_st + adjacency_matrix_row_sum_st
            vol_S_rem_st = adjacency_matrix_sum - vol_S_st
            phi_s_st = float(w_cum_sum_st/min(vol_S_st,vol_S_rem_st))
            blah_2.append(phi_s_st)

        if phi_s < phi_s_st:
            s_star = s.copy()
            pointer = False
        else:
            pointer = True
            
        t = t + 1

    sparse_cut = []
    for xy in range(0,len(vertices),1):
        if xy in s_star:
            sparse_cut.append(1)
        else:
            sparse_cut.append(0)

    sparse_cut_np = np.array(sparse_cut).reshape((100,100))
    
    return sparse_cut_np
