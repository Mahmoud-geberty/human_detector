from skimage.feature import hog
#from skimage.io import imread
import joblib,glob,os,cv2

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm, metrics
import numpy as np 
from sklearn.preprocessing import LabelEncoder

train_data = []
train_labels = []
pos_im_path = 'DATAIMAGE/positive/'
neg_im_path = 'DATAIMAGE/negative/'
hard_im_path = 'DATAIMAGE/hard_negative/'
model_path = 'models/models.dat'
# Load the positive features
for filename in glob.glob(os.path.join(pos_im_path,"*.png")):
    fd = cv2.imread(filename,0)
    fd = cv2.resize(fd,(64,128))
    fd = hog(fd,block_norm='L2',orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(2,2))
    train_data.append(fd)
    train_labels.append(1)

len1 = len(train_data)

# Load the negative features
for filename in glob.glob(os.path.join(neg_im_path,"*.jpg")):
    fd = cv2.imread(filename,0)
    fd = cv2.resize(fd,(64,128))
    fd = hog(fd,block_norm='L2',orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(2,2))
    train_data.append(fd)
    train_labels.append(0)

# Load the negative features
for filename in glob.glob(os.path.join(hard_im_path,"*.jpg")):
    fd = cv2.imread(filename,0)
    fd = cv2.resize(fd,(64,128))
    fd = hog(fd,block_norm='L2',orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(2,2))
    train_data.append(fd)
    train_labels.append(0)

len2 = len(train_data) - len1


train_data = np.float32(train_data)
train_labels = np.array(train_labels)

print('Data Prepared........')
print(f" positive samples:{len1}")
print(f" negative samples:{len2}")
print('Train Data:',len(train_data))
print('Train Labels (1,0)',len(train_labels))
print("""
Classification with SVM

""")


#model = LinearSVC(max_iter=10000,  tol=0.00000001)
model = LinearSVC(C=0.1, max_iter=4000, dual=False, tol=0.0000000001)
print('Training...... Support Vector Machine')
model.fit(train_data,train_labels)

joblib.dump(model, 'models/models.dat')
print('Model saved : {}'.format('models/models.dat'))
