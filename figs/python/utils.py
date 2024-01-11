import numpy as np
import os
from typing import List, Optional, Dict, Union


def parse_history(fdir:str)->Dict[str, Union[float, np.ndarray, List[float]]]:
    """
    Parameters:
    - fname: absolute path of dir containing the training_history.txt
    """
    fname = os.path.join(fdir, "training_history.txt")
    with open(fname, 'r') as f:
        lines = f.readlines()

    dic_train = parse_training_loss(lines)
    dic_results = parse_results(lines)
    
    return {**dic_train, ** dic_results}

def parse_training_loss(lines:List[str])->Dict[str, np.ndarray]:
    epoch = []
    loss = []
    internal = []
    external = []
    boundary = []
    time_per = []
    for line in lines:
        if line[:5] == "Iter:":
            tmp = line.split()
            epoch.append(float(tmp[1]))
            loss.append(float(tmp[3]))
            internal.append(float(tmp[5]))
            external.append(float(tmp[7]))
            boundary.append(float(tmp[9]))
            time_per.append(float(tmp[11][:-6]))
    dic_train = {
        "epoch": np.array(epoch),
        "loss": np.array(loss),
        "internal": np.array(internal),
        "external": np.array(external),
        "boundary": np.array(boundary),
        "time_per": np.array(time_per),
    }     
    return dic_train


def parse_results(lines:List[str])->Dict[str, Union[float, List[float]]]:
    dic_results = {}
    line_num = 0
    while (line_num<len(lines)):
        if lines[line_num][:3] == "End":
            dic_results["time"] = float(lines[line_num].split()[-1].replace('\n', ''))
        elif lines[line_num][:3] == "L2 ":
            dic_results["L2norm"] = float(lines[line_num].split()[-1].replace('\n', ''))
        elif lines[line_num][:3] == "H10":
            dic_results["H10norm"] = float(lines[line_num].split()[-1].replace('\n', ''))
        elif lines[line_num][:3] == "[lx":
            tmp_line = lines[line_num].split()
            dic_results["configuration"] = [
                float(tmp_line[-3][1:-1]),
                float(tmp_line[-2][:-1]),
                float(tmp_line[-1][:-2])
            ]
        elif lines[line_num][:3] == "[ta":
            tmp_line = lines[line_num].split()
            dic_results["targetU"] = [
                float(tmp_line[-3][1:-1]),
                float(tmp_line[-2][:-1]),
                float(tmp_line[-1][:-2])
            ]
        elif lines[line_num][:3] == "[dx":
            tmp_line = lines[line_num].split()
            dic_results["U"] = [
                float(tmp_line[-3][5:-1]),
                float(tmp_line[-2][:-1]),
                float(tmp_line[-1][:-2])
            ]
        line_num +=1
    return dic_results
 

if __name__ == "__main__":
    fname = "/home/shguan/dem_hyperelasticity/degraded_displacement/normalNet_displacement/"\
    "degraded_simpson_25x25x4_theta1.0_phi0.00_K5.0e+05_mu6.2e+03_layers4_leftPenalty1e+02_helth1.0_tx3.8e+04" 
    save_dir = "/home/shguan/dem_hyperelasticity/degraded_displacement/normalNet_displacement/"

    dic = parse_history(fname=fname, save_dir=save_dir)
    print()
