import numpy as np
import cv2
import os
import time


# Get patch returns a cropped portion of the image provided using the globally defined radius
# pixel is a tuple of (row, column) which is the row number and column number of the pixel in the picture
# image is a cv2 image
def get_patch(pixel, image):
	radius = 6  # Used for patch size
	diameter = 2 * radius
	# max_row, max_col, not_used = np.array(image).shape Having this was making it super slow, so just manually put
	# in the size of the images i guess
	max_row = 2592
	max_col = 3888
	if pixel[0] >= (max_row - radius):
		corner_row = max_row - (diameter + 2)
		center_row = radius + pixel[0] - (max_row - (radius + 1))
	elif pixel[0] >= radius:
		corner_row = pixel[0] - radius
		center_row = radius
	else:  # With the row coordinate being less than the radius of the patch, it has to be at the top of the image
		corner_row = 0  # meaning the row coordinate for the patch will have to be 0
		center_row = pixel[0]  # Because the pixel in question is less than the radius, the center of the patch
								# should be the same as the pixel coming in as it should be less than 11

	if pixel[1] >= (max_col - radius):
		corner_col = max_col - (diameter + 2)
		center_col = radius + pixel[1] - (max_col - (radius + 1))
	elif pixel[1] >= radius:
		corner_col = pixel[1] - radius
		center_col = radius
	else:  # With the column coordinate being less than the radius of the patch, it has to be in the left side of the
		corner_col = 0  # Image, meaning the column coordinate for the patch will have to be 0
		center_col = pixel[1]  # same as the row
	diameter += 1  # Added 1 for the center pixel
	return image[corner_row:(corner_row+diameter), corner_col:(corner_col+diameter)], (center_row, center_col)


# Returns the dominate colors in a patch, which are the average colors based upon what is the center of clusters that
# are built from the rgb values in the patch, patch is a 3d array which is 50x50x3
def get_dominate_color(patch):
	z = patch.reshape((-1, 3))
	# Set the rgb values to be floats so it can be used in the k-means function
	z = np.float32(z)
	# Create the criteria for k-means clustering, 1st: Stop kmeans when the specified accuracy is met, or when the
	# max iterations specified is met. 2nd: max iterations. 3rd: epsilon, or required accuracy
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	k = 2
	# run the K means clustering using cv2, so it can be done easily with images
	# label and center being the important returns, with label being important for producing an image to show the clusters
	# and center being useful for the NN and the producing the image to show the clusters, as its the average color of each
	# cluster. Arguments for the kmeans, 1st: imput data, 2nd: number of clusters needed, 3rd: not sure,
	# 4th: the criteria specified above, 5th: number of times to run the clustering taking the best result, 6th: flags
	ret, label, center = cv2.kmeans(z, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center)  # Center will contain the Dominant Colors in their respective color channels
	# i.e [[DCb, DCg, DCr], [DCb, DCg, DC3r]] with k = 2
	return center


# Uses a modified version of the method detailed in the article "Vision-Based Corrosion Detection Assisted by a
# Micro-Aerial Vehicle in a Vessel Inspection Application" by Ortiz et. Al.. It describes using the RGB values to
# create a texture feature vector based on the difference in color from the center pixel and the neighbors selected
# in a circle around the pixel in question. p neighbors are selected in a similar fashion to the article,
# expect instead of using bilinear interpolation, I just rounded the x and y values to get a proper index for the
# patch matrix.
def get_texture(patch, pixel):
	c_b, c_g, c_r = patch[pixel[0], pixel[1]]  # The rgb values of the center pixel
	neighbor = [[], [], []]
	if pixel[0] < 6 or pixel[1] < 6 or pixel[0] > 6 or pixel[1] >6:  # With the center not being the actual center,
		# neighbors are chosen at random
		x = np.random.choice(12, 8)  # Very Important!! This needs to go to the size of the patch,
		y = np.random.choice(12, 8)  # which is done manually, so if the size of the patch is changed this
		#  needs to be as well
	else:
		radius = 3  # Used to get the neighbors, needs to be smaller than the patch radius
		p = 8  # total number of neighbors selected
		k = np.array(list(range(1, p+1)))  # k is the index of the neighbors, 1 through p, the range function is [a, b)
		# which is why i use p+1
		a_k = [((k-1) * 2 * np.pi) / p]  # a_k is the radial transformation of the k indexes
		x = radius * np.cos(a_k)  # The x coordinate for the neighbors
		y = -radius * np.sin(a_k)  # The y coordinate for the neighbors
		x = np.round(pixel[0] + x)  # pixel is the x & y coordinate for the central pixel, which is the offset for the
		y = np.round(pixel[1] + y)  # neighbors, which is then rounded off so it is a whole number
		x = x.astype(int).flatten()
		y = y.astype(int).flatten()
	for i in range(len(x)):  # getting the rgb values for all of the neighbors
		b, g, r = patch[x[i], y[i]]
		neighbor[0].append(r)
		neighbor[1].append(g)
		neighbor[2].append(b)
	neighbor = np.array(neighbor).astype(int)
	diff = [neighbor[0] - c_r, neighbor[1] - c_g, neighbor[2] - c_b]
	pos_diff = np.array([[i > 0 for i in diff[0]], [i > 0 for i in diff[1]], [i > 0 for i in diff[2]]])
	neg_diff = np.array([[i < 0 for i in diff[0]], [i < 0 for i in diff[1]], [i < 0 for i in diff[2]]])
	return [np.sum(pos_diff[0]**2), np.sum(pos_diff[1]**2), np.sum(pos_diff[2]**2), np.sum(neg_diff[0]**2),
			np.sum(neg_diff[1]**2), np.sum(neg_diff[2]**2)]


def init(filepath):
	for image in os.walk(filepath):
		image_paths = image[2]
	os.chdir(filepath)
	images = np.array([cv2.imread(i) for i in image_paths])
	return images


if __name__ == '__main__':
	path = "c:\\users\\dakot\\Desktop\\metal scraps\\"
	start_time = time.time()
	images = init(path)
	t_time = time.time()
	print("Images loaded:" + str(t_time - start_time))
	total_time = t_time
	total_patch_time = 0
	total_color_time = 0
	total_texture_time = 0
	temp = images[0]
	for i in range(0, 2592):  # TODO: add parallelism, saving of the features
		row_time = time.time()
		row_patch_time = 0
		row_color_time = 0
		row_texture_time = 0
		for j in range(0, 3888):
			center_pixel = (i, j)
			prev_time = time.time()
			patch, pixel = get_patch(center_pixel, temp)
			current_time = time.time()
			row_patch_time += current_time - prev_time
			prev_time = time.time()
			descriptor_color = get_dominate_color(patch)
			row_color_time += time.time() - prev_time
			prev_time = time.time()
			descriptor_texture = get_texture(patch, pixel)
			row_texture_time += time.time() - prev_time
			prev_time = time.time()
		print("Row " + str(i) + " time is : " + str(prev_time - row_time) + " secs")
		print("Row patch time: " + str(row_patch_time) + "\nRow color time: " + str(row_color_time) + "\nRow texture "
				"time: " + str(row_texture_time) + "\n\n")
		total_patch_time += row_patch_time
		total_color_time += row_color_time
		total_texture_time += row_texture_time
	print("Final time: " + str(prev_time - total_time))
	print("Total patch time: " + str(total_patch_time) + "\nTotal color time: " + str(total_color_time) + "\nTotal texture "
		"time: " + str(total_texture_time))