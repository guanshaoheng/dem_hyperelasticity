from import_file import *
from Utility import cal_jacobian


class MultiLayerNet_curl(torch.nn.Module):
    def __init__(self, D_in, H, D_out, num_layers=4):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MultiLayerNet_curl, self).__init__()
        self.linear_in = torch.nn.Linear(D_in, H)
        self.linear_out = torch.nn.Linear(H, D_out)
        self.num_layers = num_layers

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(H, H) for _ in range(num_layers-2)
        ])

        torch.nn.init.constant_(self.linear_in.bias, 0.)
        torch.nn.init.constant_(self.linear_out.bias, 0.)
        torch.nn.init.normal_(self.linear_in.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.linear_out.weight, mean=0, std=0.1)
        for i in range(num_layers-2):
            torch.nn.init.constant_(self.layers[i].bias, 0.)
            torch.nn.init.normal_(self.layers[i].weight, mean=0, std=0.1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        
        x = torch.tanh(self.linear_in(x))
        for i in range(self.num_layers-2):
            x = torch.tanh(self.layers[i](x))
        x = self.linear_out(x)
        return x
    
    def forward_curl(self, x):
        y = self.forward(x)
        dydx = cal_jacobian(x, y)

        curl = torch.stack([
            dydx[:, 2, 1] - dydx[:, 1, 2], 
            dydx[:, 0, 2] - dydx[:, 2, 0], 
            dydx[:, 1, 0] - dydx[:, 0, 1]] ).permute(1, 0)

        # check the divergence of the curl is 0 or not
        # residual_ = torch.einsum("nii->n", cal_jacobian(x, curl))
        # dcurldx = cal_jacobian(x, torch.einsum("nj, n->nj", curl, x[:, 0]**2 - 2*x[:, 0]))
        # residual = torch.einsum("nii->n", dcurldx)
        return curl
    

if __name__ == "__main__":
    net = MultiLayerNet_curl(3, 20, 3)
    x = torch.randn(size=[20 ,3],requires_grad=True)
    curl = net.forward(x)
    print("xxx")
