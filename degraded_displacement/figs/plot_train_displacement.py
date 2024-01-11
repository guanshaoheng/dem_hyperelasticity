from figs.python.utils import parse_history
import os
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 18})

def main(
        file_dir:str = "../normalNet_displacement",
    ):
    files_list_raw = os.listdir(file_dir)
    files_list = []
    for file in files_list_raw:
        if file[:5]!= "degra" or file[-5:] == "_test": 
            continue
        files_list.append(file)
    del files_list_raw

    penalties = []
    for file in files_list:
        penalties.append(float(file.split('_')[-3][11:]))
    sorted_index = sorted(list(range(len(penalties))), key=lambda x: penalties[x])
    penalties = sorted(penalties)
    
    dic_list = []
    for index in sorted_index:
        dic_list.append(parse_history(os.path.join(file_dir, files_list[index])))

    plot_train_loss(penalties, dic_list)
    plot_displacement(penalties, dic_list)
    

def plot_train_loss(
        penalties, 
        dic_list):
    for index in range(len(penalties)):
        plt.semilogy(
            dic_list[index]["epoch"]/1e3, 
            dic_list[index]["loss"], 
            label="%.0e" % penalties[index], 
            linewidth=3)
    plt.legend(
        loc="upper right", 
        ncol=2, 
        fontsize=18)
    plt.xlabel("Epoch (1e3)")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.grid()
    plt.savefig("./train_loss.png", dpi=200)
    plt.close()


def plot_displacement(
        penalties, 
        dic_list):
    u = []
    for index in range(len(penalties)):
        u.append(dic_list[index]["U"][0])

    plt.plot(penalties, u, linewidth=3, color='b', marker='X', markersize='9')
    plt.plot(penalties, [1.0] * len(penalties), color='gray', linestyle='--', linewidth=3)

    plt.xlabel("Penalties")
    plt.ylabel("Displacement in x direction")
    plt.xscale('log')  # Set the x-axis to log scale
    plt.tight_layout()
    plt.grid()
    plt.savefig("./displacement.png", dpi=200)
    plt.close()



if __name__ == "__main__":
    main()
        


