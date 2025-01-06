import numpy as np

def read_cam_params(filename): 
    # Read the contents of the text file
    var_dict = {}
    with open(filename) as file:
        for line in file:
            if "=" in line and not line.startswith("#"):
                name, value = line.split("=")
                var_dict[name.strip()] = eval(value.strip())
    return var_dict

   
class Camera:
    def __init__(self, filename):
        data = read_cam_params(filename)
        self.camera_matrix = np.array([[data['fx'], 0., data['cx']],[0., data['fy'], data['cy']],[0., 0., 1.]])
        self.distcoeff = np.array([data['k1'], data['k2'], data['p1'], data['p2'], data['k3']])
        
        data = {key: value for key, value in data.items() if key not in ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'p3']}
        for key, value in data.items():
            setattr(self, key, value)
            
    def cam_matrix(self):
        return self.camera_matrix
    
    def dist_coeff(self):
        return self.distcoeff


if __name__ == '__main__':
    import sys
    # print(read_cam_params('camera_parameters/DSCRX0M2-3171074-20211201-chessboard.txt'))
    camRX0 = Camera(sys.argv[1])
    print(camRX0.cam_matrix())
    print(camRX0.dist_coeff())
