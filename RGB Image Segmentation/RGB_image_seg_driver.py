
"""
Description: Driver code which impliments functionality within the RGB_image_seg.py module.
"""
import RGB_image_seg as func
import time
import matplotlib.pyplot as plt
import os


file_path_input = input("Enter the the file path where the .png image is stored\n")
assert os.path.exists(file_path_input), "I did not find the file at, "+str(file_path_input)
print("We found your file!")

start_time = time.time()

gaus_blur = func.down_sampled_image(file_path = file_path_input)

normalise_cordinates = func.normalise_gaus_blur_cordinates(gaus_blur)
adjacency_matrix,no_edges,average_deg = func.construct_adjacency_matrix(normalise_cordinates)
heatmap_matrix = func.construct_heatmap_matrix(adjacency_matrix)
sparse_cut = func.construct_sparse_cut(adjacency_matrix = adjacency_matrix,heatmap_matrix = heatmap_matrix)

plt.figure(1)
plt.imshow(gaus_blur)
plt.figure(2)
plt.imshow(heatmap_matrix)
plt.figure(3)
plt.imshow(sparse_cut)
plt.show()


end_time = time.time()
print("Elapsed time was %g seconds" % (end_time - start_time))
print("Number of edges of the constructed graph: ",no_edges)
print("Average degree of the constructed graph: ",average_deg)
