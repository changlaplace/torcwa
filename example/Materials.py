import numpy as np
import torch
from scipy.interpolate import interp1d

# class aSiH(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, wavelength, dl = 0.005):
#         # open material data
#         open_name = 'Materials_data/aSiH.txt'
#         f = open(open_name)
#         data = f.readlines()
#         f.close()
#         nk_data = []
#         for i in range(len(data)):
#             _lamb0, _n, _k = data[i].split()
#             nk_data.append([float(_lamb0), float(_n), float(_k)])
#         nk_data = np.array(nk_data)

#         n_interp = interp1d(nk_data[:,0],nk_data[:,1],kind='cubic')
#         k_interp = interp1d(nk_data[:,0],nk_data[:,2],kind='cubic')

#         wavelength_np = wavelength.detach().cpu().numpy()

#         if wavelength_np < nk_data[0,0]:
#             nk_value = nk_data[0,1]+1.j*nk_data[0,2]
#         elif wavelength_np > nk_data[-1,0]:
#             nk_value = nk_data[-1,1]+1.j*nk_data[-1,2]
#         else:
#             nk_value = n_interp(wavelength_np)+1.j*k_interp(wavelength_np)

#         if wavelength_np-dl < nk_data[0,0]:
#             nk_value_m = nk_data[0,1]+1.j*nk_data[0,2]
#         elif wavelength_np-dl > nk_data[-1,0]:
#             nk_value_m = nk_data[-1,1]+1.j*nk_data[-1,2]
#         else:
#             nk_value_m = n_interp(wavelength_np-dl)+1.j*k_interp(wavelength_np-dl)

#         if wavelength_np+dl < nk_data[0,0]:
#             nk_value_p = nk_data[0,1]+1.j*nk_data[0,2]
#         elif wavelength_np+dl > nk_data[-1,0]:
#             nk_value_p = nk_data[-1,1]+1.j*nk_data[-1,2]
#         else:
#             nk_value_p = n_interp(wavelength_np+dl)+1.j*k_interp(wavelength_np+dl)

#         ctx.dnk_dl = (nk_value_p - nk_value_m) / (2*dl)
        
#         return torch.tensor(nk_value,dtype=torch.complex128 if ((wavelength.dtype is torch.float64) or\
#             (wavelength.dtype is torch.complex128)) else torch.complex64, device=wavelength.device)

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad = 2*torch.real(torch.conj(grad_output)*ctx.dnk_dl)
#         return grad
    
class DispersiveMaterial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, wavelength, filename, dl=0.005):
        data = np.loadtxt(filename)
        lamb = data[:,0]
        n_data = data[:,1]
        k_data = data[:,2]

        n_interp = interp1d(lamb, n_data, kind='cubic')
        k_interp = interp1d(lamb, k_data, kind='cubic')

        wl = wavelength.detach().cpu().numpy()

        def nk_at(w):
            if w < lamb[0]:
                return n_data[0] + 1j*k_data[0]
            elif w > lamb[-1]:
                return n_data[-1] + 1j*k_data[-1]
            else:
                return n_interp(w) + 1j*k_interp(w)

        nk0 = nk_at(wl)
        nk_p = nk_at(wl + dl)
        nk_m = nk_at(wl - dl)

        ctx.dnk_dl = (nk_p - nk_m) / (2*dl)

        dtype = torch.complex128 if (
            wavelength.dtype in [torch.float64, torch.complex128]
        ) else torch.complex64

        return torch.tensor(nk0, dtype=dtype, device=wavelength.device)

    @staticmethod
    def backward(ctx, grad_output):
        grad = 2 * torch.real(torch.conj(grad_output) * ctx.dnk_dl)
        return grad, None, None
    

class aSiH:
    @staticmethod
    def apply(wavelength):
        return DispersiveMaterial.apply(
            wavelength, "Materials_data/aSiH.txt"
        )


class Al:
    @staticmethod
    def apply(wavelength):
        return DispersiveMaterial.apply(
            wavelength, "Materials_data/Al.txt"
        )

# 1.46**2
class SiO2:
    @staticmethod
    def apply(wavelength):
        return DispersiveMaterial.apply(
            wavelength, "Materials_data/SiO2.txt"
        )


class TiO2:
    @staticmethod
    def apply(wavelength):
        return DispersiveMaterial.apply(
            wavelength, "Materials_data/TiO2.txt"
        )