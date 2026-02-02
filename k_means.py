import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()


dataset = data.data

k = 7

def k_means(dataset, k):

    #mean normalization
    mean = np.mean(dataset, axis = 0)
    normalized_dataset = dataset - mean

    #scaling data
    data_range = np.amax(normalized_dataset, axis=0) - np.amin(normalized_dataset, axis = 0)
    standard_dataset = normalized_dataset/data_range

    # initialization of k centroids (to random training points)
    centroids = []

    i = 0
    for i in range(k):
        random_ind = np.random.randint(0,len(standard_dataset))
        centroids.append(standard_dataset[random_ind])
    distortions = []

    stop = 0
    while(stop != 1):

        #cluster assignment
        cluster_assignments = []

        for i in range(len(standard_dataset)):
            cur_min_dist = np.linalg.norm(standard_dataset[i]-centroids[0])
            cur_min_ind = 0
            for j in range(1,k):

                dist = np.linalg.norm(standard_dataset[i]-centroids[j])
                if dist < cur_min_dist:
                    cur_min_dist = dist
                    cur_min_ind = j
            cluster_assignments.append(cur_min_ind)


        #calculate distortion
        distortion = 0
        for i in range(len(standard_dataset)):
            distortion = distortion + np.linalg.norm(standard_dataset[i]-centroids[cluster_assignments[i]])**2
        distortion = distortion/len(standard_dataset)
        distortions.append(distortion)
        #print(distortion)


        #move centroid
        for i in range(k):
            assigned = np.array([])
            for j in range(len(cluster_assignments)):
                if i == cluster_assignments[j]:
                    if len(assigned) > 0:
                        assigned = np.append(assigned, np.array([standard_dataset[j]]), axis = 0)
                    else:
                        assigned = np.array([standard_dataset[j]])
            if len(assigned) > 0:
                new_centroid = assigned.mean(axis=0)

            #check stop condition
            if np.linalg.norm(centroids[i] - np.array(new_centroid)) == 0 and i == 0:
                stop = 1

            elif np.linalg.norm(centroids[i] - np.array(new_centroid)) > 0:
                stop = 0


            centroids[i] = new_centroid



    #cluster assignment
    cluster_assignments = []

    for i in range(len(standard_dataset)):
        cur_min_dist = np.linalg.norm(standard_dataset[i]-centroids[0])
        cur_min_ind = 0
        for j in range(1,k):
            dist = np.linalg.norm(standard_dataset[i]-centroids[j])
            if dist < cur_min_dist:
                cur_min_dist = dist
                cur_min_ind = j
        cluster_assignments.append(cur_min_ind)


    #calculate distortion

    distortion = 0
    for i in range(len(standard_dataset)):
        distortion = distortion + np.linalg.norm(standard_dataset[i]-centroids[cluster_assignments[i]])**2
    distortion = distortion/len(standard_dataset)
    distortions.append(distortion)

    return [centroids, cluster_assignments, distortions]




#plot k vs distortion
y = []

for k in range(2,30):
    print(k)
    print(k_means(dataset, k)[2][-1])
    y.append(k_means(dataset, k)[2][-1]) #add distortion values to the list y


k = list(range(2,30))

plt.plot(k, y, "-o")
plt.xlabel("k")
plt.ylabel("Distortion")
plt.title("k vs Distortion")
plt.show()





















