import numpy as np
from matplotlib import pyplot as plt


# 计算球形积分
def func_to_be_integrated(
        N_tmp: np.ndarray, E_r: np.ndarray,
        be: float=1.0 ):
    
    v = np.exp(be*(np.dot(N_tmp, E_r))**2)
    return v

def norm_vec(theta, psi):
    return np.array([
        np.sin(theta) * np.cos(psi),
        np.sin(theta) * np.sin(psi),
        np.cos(theta)])


def integration(theta_fix: float, psi_fix: float, num_pieces:int=5 ):
    dh = 0.5*np.pi/num_pieces
    num_psi = num_pieces*4+1  # 360 度
    num_theta = num_pieces*4+1  # 90 度
    theta = np.linspace(0., 2.0 * np.pi, num_theta)
    psi = np.linspace(0., 2.0 * np.pi, num_psi)
    sum = 0.
    E_r = norm_vec(theta_fix, psi_fix)
    for i in theta[:-1]:
        for j in psi[:-1]:
            theta_tmp = i+0.5*dh
            psi_tmp = j+0.5*dh
            # NOTE: use the mid points of the square
            v = func_to_be_integrated(norm_vec(theta_tmp, psi_tmp), E_r)
            sum += v*dh**2*np.abs(np.sin(theta_tmp))

    # sum /=9.19
    print('Num_pieces %d integration: %.4e' % (num_pieces, sum))
    return sum


def main():
    
    psi_fix=0.

    num_pieces_list = [5]  #[3, 5, 10, 20, 40, 50, 100, 200]
    theta_list = np.linspace(0., np.pi*0.5, 10)
    sum_list = []
    for i in num_pieces_list:
        for j in theta_list:
            sum_list.append(integration(j, psi_fix, num_pieces=i))

    print(sum_list)
    # plt.semilogx(num_pieces_list, sum_list)
    plt.plot(theta_list, sum_list)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    main() 