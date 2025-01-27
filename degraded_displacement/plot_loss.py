from plot_configuration import *


def read_file(
        file_dir: str):
    f = open(file_dir + "/training_history.txt", 'r')
    data = f.readlines()
    f.close()
    epoch_list = []
    loss_list = []
    internal_list = []
    external_list = []
    boundary_penalty = []
    epoch_tmp = 1
    for line in data:
        tmp = line.split(' ')
        if tmp[0] != "Iter:": break
        epoch_list.append(epoch_tmp)
        loss_list.append(float(tmp[3]))
        internal_list.append(float(tmp[5]))
        external_list.append(float(tmp[7]))
        boundary_penalty.append(float(tmp[9]))
        epoch_tmp += 1; 
    return {
        "epoch": np.array(epoch_list),
        "Loss": np.array(loss_list),
        "Internal": np.array(internal_list),
        "External": np.array(external_list),
        "Boundary": np.array(boundary_penalty),
        }

def main():
    plt.figure()

    # 绘制曲线
    for key in tmp_dic:
        if key != "epoch":
            if key == "Internal":
                # plt.semilogy(tmp_dic["epoch"], tmp_dic[key], label=key, linestyle="--")
                plt.plot(tmp_dic["epoch"], tmp_dic[key], label=key, linestyle="--")
            elif key == "Boundary":
                continue
            else:
                # plt.semilogy(tmp_dic["epoch"], tmp_dic[key], label=key)
                plt.plot(tmp_dic["epoch"], tmp_dic[key], label=key)

    # 添加图例
    plt.legend()

    # 添加标题和标签
    plt.xlabel("Epoch")
    # plt.ylabel("Loss")

    # 显示网格线
    plt.grid(True, linestyle='--')

    plt.tight_layout()

    fig_name = os.path.join(fname, "loss.png")
    plt.savefig(fig_name, dpi=300)
    plt.show()
    return 


if __name__ == "__main__":
    fname = "degraded_simpson_beam20x20x4_theta0_phi0_helth0.0_K6e+04_mu6e+04_iter1000000"
    tmp_dic = read_file(file_dir=fname)
    main()
