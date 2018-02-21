
# coding: utf-8

# In[ ]:


import numpy as np
import math
import numpy.linalg as la
def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
lambda_ang = 2
lambda_lr  = 0.5
lambda_skc = 0.5 #чувствительность от потери джоинта в длинах его сегмента
lambda_ac = 5
lambda_rot = 0.5
alfa_1 = 0.07 # penalty_1
alfa_2 = 0.07 # penalty_2 

def angle(v1, v2):
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)

def joint_similarity(joint_x,joint_y):    
    #joint_x
    x_coords_x = [x[0] for x in joint_x]
    y_coords_x = [y[1] for y in joint_x] 
    #считаем длины сегментов joint_x 
    left_length_x = ((x_coords_x[1] - x_coords_x[0])**2 + (y_coords_x[1] - y_coords_x[0])**2)**0.5
    right_lentgh_x = ((x_coords_x[1] - x_coords_x[2])**2 + (y_coords_x[1] - y_coords_x[2])**2)**0.5
    #отношение длин сегментов joint_x
    gamma_x = left_length_x/right_lentgh_x
    #угол joint_x
    point_a_x = [x_coords_x[0]-x_coords_x[1], y_coords_x[0]-y_coords_x[1]]
    point_b_x = [x_coords_x[2]-x_coords_x[1], y_coords_x[2]-y_coords_x[1]]
    ang_a_x = np.arctan2(*point_a_x[::-1])
    ang_b_x = np.arctan2(*point_b_x[::-1])
    angle_x = np.rad2deg((ang_b_x - ang_a_x) % (2 * np.pi))*np.pi/180 
    
    #joint_y
    x_coords_y = [x[0] for x in joint_y]
    y_coords_y = [y[1] for y in joint_y] 
    #считаем длины сегментов joint_y 
    left_length_y = ((x_coords_y[1] - x_coords_y[0])**2 + (y_coords_y[1] - y_coords_y[0])**2)**0.5
    right_lentgh_y = ((x_coords_y[1] - x_coords_y[2])**2 + (y_coords_y[1] - y_coords_y[2])**2)**0.5
    #отношение длин сегментов joint_y
    gamma_y = left_length_y/right_lentgh_y
    #угол joint_y
    point_a_y = [x_coords_y[0]-x_coords_y[1], y_coords_y[0]-y_coords_y[1]]
    point_b_y = [x_coords_y[2]-x_coords_y[1], y_coords_y[2]-y_coords_y[1]]
    ang_a_y = np.arctan2(*point_a_y[::-1])
    ang_b_y = np.arctan2(*point_b_y[::-1])
    angle_y = np.rad2deg((ang_b_y - ang_a_y) % (2 * np.pi))*np.pi/180 
    
    #rotation_coefficient
    left_vector_x = [x_coords_x[0] - x_coords_x[1],y_coords_x[0] - y_coords_x[1]]
    right_vector_x =[x_coords_x[2] - x_coords_x[1],y_coords_x[2] - y_coords_x[1]]
    rot_vector_x = [left_vector_x[0] + right_vector_x[0], left_vector_x[1] + right_vector_x[1]]
    
    left_vector_y = [x_coords_y[0] - x_coords_y[1],y_coords_y[0] - y_coords_y[1]]
    right_vector_y =[x_coords_y[2] - x_coords_y[1],y_coords_y[2] - y_coords_y[1]]
    rot_vector_y = [left_vector_y[0] + right_vector_y[0], left_vector_y[1] + right_vector_y[1]]
    rotation_angle = angle(rot_vector_x, rot_vector_y)
    #print rotation_angle , "rotation_angle "
    
    segment_length_ratio = math.exp(lambda_lr * (1-min(gamma_x/gamma_y, gamma_y/gamma_x)))
    
    segment_angle_ratio = math.exp(-lambda_ang * abs(angle_x-angle_y))
    segment_rotation_ratio = math.exp(-lambda_rot * abs(rotation_angle))
    #print segment_rotation_ratio, "segment_rotation_ratio"
    #print segment_length_ratio, "segment_length_ratio"
    #print segment_angle_ratio, "segment_angle_ratio"
    #print 1+math.exp(lambda_lr)+math.exp(lambda_rot*np.pi), "normalization term"
    #print 2*np.pi, "np.pi"
    return (segment_length_ratio*segment_angle_ratio*segment_rotation_ratio)
    #return segment_length_ratio*segment_angle_ratio
    #return 1.0


def sim_matrix(chain1, chain2):
    loop_chain1=[]
    loop_chain1.append(chain1[len(chain1)-1])
    for j in range(len(chain1)):
        loop_chain1.append(chain1[j])
    loop_chain1.append(chain1[0])
    #массив джоинтов у chain1
    seq_of_joints1 = []
    for j in range(len(loop_chain1)-2):
        seq_of_joints1.append((loop_chain1[j], loop_chain1[j+1],loop_chain1[j+2]))
    #looping chain1
    loop_chain2=[]
    loop_chain2.append(chain2[len(chain2)-1])
    for j in range(len(chain2)):
        loop_chain2.append(chain2[j])
    loop_chain2.append(chain2[0])
    #массив джоинтов у chain2
    seq_of_joints2 = []
    for j in range(len(loop_chain2)-2):
        seq_of_joints2.append((loop_chain2[j], loop_chain2[j+1],loop_chain2[j+2]))
    H = np.zeros((len(chain1)+1, len(chain2)+1))
    #print len(chain1)
    #print  seq_of_joints2, " seq_of_joints2"
    #print  seq_of_joints1, " seq_of_joints1"

    for i in range(len(chain1)):
        #print "i=",i, "seq_of_joints1[i]=", seq_of_joints1[i]
        for j in range(len(chain2)):
            H[i+1,j+1]=joint_similarity(seq_of_joints1[i], seq_of_joints2[j])
            #print "j=",j, "seq_of_joints1[j]=", seq_of_joints2[j],H[i+1,j+1],"H[i,j]"
    #print H, "sim_matrix !"
    #print pd.DataFrame(H)
    return H 

def pen_matrix(chain):
    loop_chain=[]
    loop_chain.append(chain[len(chain)-1])
    for j in range(len(chain)):
        loop_chain.append(chain[j])
    loop_chain.append(chain[0])
    x_coords = [x[0] for x in loop_chain]
    y_coords = [y[1] for y in loop_chain] 
    lengths_ratios=[]
    angles=[]
    penalties = []
    for j in range(len(loop_chain)-2):
        left_length = ((x_coords[j] - x_coords[j+1])**2 + (y_coords[j] - y_coords[j+1])**2)**0.5
        right_lentgh = ((x_coords[j+2] - x_coords[j+1])**2 + (y_coords[j+2] - y_coords[j+1])**2)**0.5        
        point_a = [x_coords[j]-x_coords[j+1], y_coords[j]-y_coords[j+1]]
        point_b = [x_coords[j+2]-x_coords[j+1], y_coords[j+2]-y_coords[j+1]]
        ang_a = np.arctan2(point_a[1],point_a[0])
        ang_b = np.arctan2(point_b[1], point_b[0])
        angle = np.rad2deg((ang_b - ang_a) % (2 * np.pi))
        pen = 1 - math.exp(-abs(np.pi - angle)) + lambda_skc*(left_length+right_lentgh)/2
        penalties.append(pen)
    #print "penalties", penalties 
    return penalties

def chain_matching_score(chain1, chain2): 
        H = np.zeros((len(chain1) + 1, len(chain2) + 1))
        #corr_chain1 = to_corr_chain(chain1)
        #corr_chain2 = to_corr_chain(chain2)
        s_matrix  = sim_matrix(chain1, chain2)
        p_matrix1 = pen_matrix(chain1)
        p_matrix2 = pen_matrix(chain2)
        max_score = 0 
        max_pos   = None
        numb_matches = 1
        for i in range(len(chain1)):
            for j in range(len(chain2)):
                H[i+1,j+1] = max(H[i,j]+s_matrix[i+1,j+1],
                                 H[i,j+1]- alfa_1*p_matrix1[i],H[i+1,j]- alfa_2*p_matrix2[j])
                #print "H[i-1,j-1]=", H[i,j],'!',"s_matrix[i+1,j+1]=", s_matrix[i+1,j+1], "!"
                #candidates_to = np.asarray([H[i,j]+s_matrix[i+1,j+1],H[i,j+1]- p_matrix1[i],H[i+1,j]- p_matrix2[j]])
                #print "candidates_to", candidates_to,'!', "H[i+1,j+1]", H[i+1,j+1],'!'   
                if  H[i+1,j+1] > max_score:
                    max_score = H[i+1,j+1]
                    #print max_score,s_matrix[i+1,j+1]
                    max_pos   = (i+1, j+1)
        #print pd.DataFrame(H) 
        
        #traceback
        chains_similarity = max_score
        #print "chains_similarity", chains_similarity, "!"
        next_value = 0
        candidates=[]
        #i=0
        #j=0
        while True:   
            next_value = max(H[max_pos[0], max_pos[1]-1], H[max_pos[0]-1, max_pos[1]-1], H[max_pos[0]-1, max_pos[1]])
            candidates = np.asarray([H[max_pos[0], max_pos[1]-1], H[max_pos[0]-1, max_pos[1]-1], H[max_pos[0]-1, max_pos[1]]])
            #print "candidates_for_back", candidates,'!', "next_value",next_value,'!'
            if candidates.argmax() == 0:
                max_pos = [max_pos[0], max_pos[1]-1]
                #print "skip"
            if candidates.argmax() == 1:
                max_pos = [max_pos[0]-1, max_pos[1]-1]
                #print "match!"
                numb_matches = numb_matches + 1
            if candidates.argmax() == 2:
                max_pos = [max_pos[0]-1, max_pos[1]]
                #print "skip"
            if max_pos[1] == 0:
                break
            if max_pos[0] == 0:
                break
            max_length=max(len(chain1),len(chain2))
            #print "max_length", max_length, "!!!"
            #print "новая max_pos", max_pos, "max_pos[0]-", max_pos[0],'max_pos[1]-', max_pos[1]
            chains_similarity = chains_similarity + next_value 
            #print max_length, "max_length"
            #print "chains_similarity", chains_similarity, "!"
        
        chains_similarity = chains_similarity/(max_length*(max_length+1)/2.0)
        #print pd.DataFrame(H)
        return chains_similarity, numb_matches

def clean_cos(cos_angle):
    return min(1,max(cos_angle,-1))

def sum_angles_of(points):
    x_coords = [x[0] for x in points]
    y_coords = [y[1] for y in points]    
    chain_len = len(points)
    centroid_x = sum(x_coords)/chain_len #нашли первую координату барицентра
    centroid_y = sum(y_coords)/chain_len #вторую
    tensor=[]
    sum_angles = 0
    for p in points:
        t = [(centroid_x-p[0]), (centroid_y-p[1])] 
        tensor.append(t)
    for t in range(len(tensor)-1):
        n_angle = angle(tensor[t], tensor[t+1])
        sum_angles = sum_angles + n_angle
    return sum_angles
"""
        tensor.append(t)
    for t in range(len(tensor)-1):
        dot_prod = dot(tensor[t], tensor[t+1])
    #Считаем длины сегментов
        length_seg_1 = dot(tensor[t], tensor[t])**0.5
        length_seg_2 = dot(tensor[t+1], tensor[t+1])**0.5
    #Находим косинусы между сегментами
        cos_ = 1.0*dot_prod/length_seg_1/length_seg_2
        cos_ = clean_cos(cos_)
    #Находим угол
        angle = math.acos(cos_)
        sum_angles = sum_angles + angle
"""
    
def Global_Angle_Consistency(chain1, chain2): 
    sum_angles1 = sum_angles_of(chain1)*np.pi/180 
    #print "sum_angles1=", sum_angles1, "!"
    sum_angles2 = sum_angles_of(chain2)*np.pi/180 
    #print "sum_angles2=", sum_angles2, "!"
    number_of_matches = chain_matching_score(chain1, chain2)[1]
    #print "number_of_matches=", number_of_matches, "!"
    angle_consistency = math.exp(-lambda_ac*abs(sum_angles1 - sum_angles2)/number_of_matches)
    #angle_consistency = math.exp(-lambda_ac*abs(sum_angles1 - sum_angles2))
    return angle_consistency
def distt(chain1, chain2):
    distance = Global_Angle_Consistency(chain1, chain2) * chain_matching_score(chain1, chain2)[0]
    #print Global_Angle_Consistency(chain1, chain2),"Global_Angle_Consistency",chain_matching_score(chain1, chain2)[0],"chain_matching_score(chain1, chain2)[0]"
    #distance = chain_matching_score(chain1, chain2)[0]
    return distance

