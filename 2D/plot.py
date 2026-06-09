import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ''))

# Import utilities
from utilities.losses import LpLoss

def set_size(width, fraction=1, subplots=(1, 1), height_add=0):
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1]) + height_add
    return (fig_width_in, fig_height_in)

# Setup formatting (no LaTeX dependency)
plot_fonts = {
    "text.usetex": False,
    "font.family": "sans-serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}
plt.rcParams.update(plot_fonts)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using {device} for plotting")

from models.fno import FNO2dMultiGoal, FNO2d
from models.dafno import DAFNO2dMultiGoal
from models.pno import DEEPNORM2dMultiGoal

model_classes = [FNO2dMultiGoal(4,1, 8,8,16),
                 DAFNO2dMultiGoal(4, 8,8, 16),
                 DEEPNORM2dMultiGoal(4, 8,8,16),
                 DEEPNORM2dMultiGoal(4, 8,8,16)]
model_classes_SDF = [FNO2d(in_channels=1, out_channels=1, width=16, modes=(8, 8), num_blocks=4)]

model_save_paths = ["FNO", "DAFNO", "PNO", "PNOwPINN"]
savepathsSDF = ["FNOSDF"]

modelFNO = model_classes[0]
modelFNO.load_state_dict(torch.load("./results/" + model_save_paths[0] + "/best_model.pt", map_location=device, weights_only=True))
modelDAFNO = model_classes[1]
modelDAFNO.load_state_dict(torch.load("./results/" + model_save_paths[1] + "/best_model.pt", map_location=device, weights_only=True))
modelPNO = model_classes[2]
modelPNO.load_state_dict(torch.load("./results/" + model_save_paths[2] + "/best_model.pt", map_location=device, weights_only=True))
modelPNOwPINN = model_classes[3]
modelPNOwPINN.load_state_dict(torch.load("./results/" + model_save_paths[3] + "/best_model.pt", map_location=device, weights_only=True))
modelSDF = model_classes_SDF[0]
modelSDF.load_state_dict(torch.load("./results/"+savepathsSDF[0]+"/best_model.pt", map_location=device, weights_only=True))

for m in [modelFNO, modelDAFNO, modelPNO, modelPNOwPINN, modelSDF]:
    m.eval()
    m.to(device)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x, goals=None):
        # x is [B, H, W, 1]
        x_perm = x.permute(0, 3, 1, 2)
        if goals is not None:
            out = self.model(x_perm, goals)
        else:
            out = self.model(x_perm)
        # out is [B, 1, H, W]
        out_perm = out.permute(0, 2, 3, 1)
        return out_perm

wrappedFNO = ModelWrapper(modelFNO)
wrappedDAFNO = ModelWrapper(modelDAFNO)
wrappedPNO = ModelWrapper(modelPNO)
wrappedPNOwPINN = ModelWrapper(modelPNOwPINN)
wrappedSDF = ModelWrapper(modelSDF)

def smooth_chi(mask, dist, smooth_coef):
    return torch.mul(torch.tanh(dist * smooth_coef), (mask - 0.5)) + 0.5

def load_dataset(filepath, trainDataCount, testDataCount, batch_size=2):
    dirPath = './dataset/' + filepath +"/"
    mask = np.load(dirPath + 'mask.npy')
    mask = torch.tensor(mask, dtype=torch.float).to(device)
    dist_in = np.load(dirPath + 'dist_in.npy')
    dist_in = torch.tensor(dist_in, dtype=torch.float).to(device)
    goals = np.load(dirPath + 'goal.npy')
    goals = torch.tensor(goals, dtype=torch.int).to(device)
    input = smooth_chi(mask, dist_in, 5)
    output = np.load(dirPath + 'output.npy')
    output = torch.tensor(output, dtype=torch.float).to(device)
    
    nData = len(input)
    nTrain = int(nData * trainDataCount)
    nTest = int(nData*testDataCount)
    mask_test = mask[-nTest:]
    goals_test = goals[-nTest:]
    chi_test = input[-nTest:]
    y_test = output[-nTest:]
    
    mask_test = mask_test.reshape(nTest, input.shape[1], input.shape[2], 1)
    goals_test = goals_test.reshape(nTest, 2)
    chi_test = chi_test.reshape(nTest, input.shape[1], input.shape[2], 1)
    y_test = y_test.reshape(nTest, input.shape[1], input.shape[2], 1)
            
    testData = DataLoader(TensorDataset(mask_test, chi_test, goals_test, y_test), batch_size=batch_size, shuffle=False)
    return None, testData

# Need mask functions
mask_func_arr = [lambda a: 1, lambda a: a, lambda a: a, lambda a: a]
def rel_error(x, y):
    res=np.zeros(x.shape)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i][j]==0:
                res[i][j] == 0
            else:
                res[i][j] = abs(x[i][j] -y[i][j])/y[i][j]
    return res

def plot_paper_fig(data256, data512, data1024, model, model2, mask_func, example_num, filename):
    set_ticks = True
    height_add = set_size(469, 0.99, (2, 4))[1]
    fig = plt.figure(figsize= set_size(469, 0.99, (2, 4),  height_add=0.8))
    subfigs = fig.subfigures(1, 2, wspace=0.05, width_ratios=[2.5,7.5], height_ratios=[1])
    
    axesLeft = subfigs[0].subplots(2, 1)
    axesRight = subfigs[1].subplots(2, 3)
  # Get vmin, vmax
    vmin_top = np.inf
    vmax_top = -np.inf
    vmin_bottom = 0
    vmax_bottom = -np.inf
    loss_func = LpLoss(d=2, p=2)
    with torch.no_grad():
        for mask, chi,goals, y in data256:
            chi = smooth_chi(mask, model2(mask), 5)
            out = model(chi, goals)
            out = out*mask_func(mask)
        if vmin_top > np.min(out[example_num].detach().cpu().numpy()):
            vmin_top = np.min(out[example_num].detach().cpu().numpy())
        if vmin_top > np.min(y[example_num].detach().cpu().numpy()):
            vmin_top = np.min(y[example_num].detach().cpu().numpy())
        if vmax_top < np.max(out[example_num].detach().cpu().numpy()):
            vmax_top = np.max(out[example_num].detach().cpu().numpy())
        if vmax_top < np.max(y[example_num].detach().cpu().numpy()):
            vmax_top = np.max(y[example_num].detach().cpu().numpy())
        y = y[example_num].detach().cpu().numpy()
        if vmax_bottom < np.max(rel_error(out[example_num].detach().cpu().numpy(), y)):
            vmax_bottom = np.max(rel_error(out[example_num].detach().cpu().numpy(), y))
        vmin_top = 0
        vmax_bottom = 5

    with torch.no_grad():
        for idx, (mask, chi,goals, y) in enumerate(data256):
            chi = smooth_chi(mask, model2(mask), 5)
            out = model(chi, goals)
            out = out*mask_func(mask)
            if idx == 29:
                break

        axesRight[1][0].imshow(out[example_num].detach().cpu().numpy(), vmin=vmin_top, vmax=vmax_top, origin="lower")
        axesRight[1][0].plot(goals[example_num][0].detach().cpu().numpy(), goals[example_num][1].detach().cpu().numpy(), 'ro',markersize=2)
        # axesRight[0][0].set_title("$256\times256")
        axesRight[1][0].set_xticks([], minor=True)
        axesRight[1][0].set_xticks([])
        axesRight[1][0].set_yticks([], minor=True)
        axesRight[1][0].set_yticks([])
        y = y[example_num].detach().cpu().numpy()
        axesRight[0][0].imshow(y, vmin=vmin_top, vmax=vmax_top, origin="lower")
        axesRight[0][0].plot(goals[example_num][0].detach().cpu().numpy(), goals[example_num][1].detach().cpu().numpy(), 'ro',markersize=2)
        axesRight[0][0].set_xticks([], minor=True)
        axesRight[0][0].set_xticks([])
        axesRight[0][0].set_yticks([], minor=True)
        axesRight[0][0].set_yticks([])

    with torch.no_grad():
        for mask, chi,goals, y in data512:
            chi = smooth_chi(mask, model2(mask), 5)
            out = model(chi, goals)
            out = out*mask_func(mask)
        if vmin_top > np.min(out[example_num].detach().cpu().numpy()):
            vmin_top = np.min(out[example_num].detach().cpu().numpy())
        if vmin_top > np.min(y[example_num].detach().cpu().numpy()):
            vmin_top = np.min(y[example_num].detach().cpu().numpy())
        if vmax_top < np.max(out[example_num].detach().cpu().numpy()):
            vmax_top = np.max(out[example_num].detach().cpu().numpy())
        if vmax_top < np.max(y[example_num].detach().cpu().numpy()):
            vmax_top = np.max(y[example_num].detach().cpu().numpy())
        y = y[example_num].detach().cpu().numpy()
        if vmax_bottom < np.max(rel_error(out[example_num].detach().cpu().numpy(), y)):
            vmax_bottom = np.max(rel_error(out[example_num].detach().cpu().numpy(), y))
        vmin_top = 0
        vmax_bottom = 5

    with torch.no_grad():
        for idx, (mask, chi,goals, y) in enumerate(data512):
            chi = smooth_chi(mask, model2(mask), 5)
            out = model(chi, goals)
            out = out*mask_func(mask)
            if idx == 29:
                break

        axesRight[1][1].imshow(out[example_num].detach().cpu().numpy(), vmin=vmin_top, vmax=vmax_top, origin="lower")
        axesRight[1][1].plot(goals[example_num][0].detach().cpu().numpy(), goals[example_num][1].detach().cpu().numpy(), 'ro',  markersize=2)
        # axesRight[0][0].set_title("$256\times256")
        axesRight[1][1].set_xticks([], minor=True)
        axesRight[1][1].set_xticks([])
        axesRight[1][1].set_yticks([], minor=True)
        axesRight[1][1].set_yticks([])
        y = y[example_num].detach().cpu().numpy()
        axesRight[0][1].imshow(y, vmin=vmin_top, vmax=vmax_top, origin="lower")
        axesRight[0][1].plot(goals[example_num][0].detach().cpu().numpy(), goals[example_num][1].detach().cpu().numpy(), 'ro', markersize=2)
        axesRight[0][1].set_xticks([], minor=True)
        axesRight[0][1].set_xticks([])
        axesRight[0][1].set_yticks([], minor=True)
        axesRight[0][1].set_yticks([])


    with torch.no_grad():
        for mask, chi,goals, y in data1024:
            chi = smooth_chi(mask, model2(mask), 5)
            out = model(chi, goals)
            out = out*mask_func(mask)
        if vmin_top > np.min(out[example_num].detach().cpu().numpy()):
            vmin_top = np.min(out[example_num].detach().cpu().numpy())
        if vmin_top > np.min(y[example_num].detach().cpu().numpy()):
            vmin_top = np.min(y[example_num].detach().cpu().numpy())
        if vmax_top < np.max(out[example_num].detach().cpu().numpy()):
            vmax_top = np.max(out[example_num].detach().cpu().numpy())
        if vmax_top < np.max(y[example_num].detach().cpu().numpy()):
            vmax_top = np.max(y[example_num].detach().cpu().numpy())
        y = y[example_num].detach().cpu().numpy()
        if vmax_bottom < np.max(rel_error(out[example_num].detach().cpu().numpy(), y)):
            vmax_bottom = np.max(rel_error(out[example_num].detach().cpu().numpy(), y))
        vmin_top = 0
        vmax_bottom = 5

    with torch.no_grad():
        for idx, (mask, chi,goals, y) in enumerate(data1024):
            chi = smooth_chi(mask, model2(mask), 5)
            out = model(chi, goals)
            out = out*mask_func(mask)
            if idx == 29:
                break

        axesRight[1][2].imshow(out[example_num].detach().cpu().numpy(), vmin=vmin_top, vmax=vmax_top, origin="lower")
        axesRight[1][2].plot(goals[example_num][0].detach().cpu().numpy(), goals[example_num][1].detach().cpu().numpy(), 'ro',  markersize=2)
        # axesRight[0][0].set_title("$256\times256")
        axesRight[1][2].set_xticks([], minor=True)
        axesRight[1][2].set_xticks([])
        axesRight[1][2].set_yticks([], minor=True)
        axesRight[1][2].set_yticks([])
        y = y[example_num].detach().cpu().numpy()
        im= axesRight[0][2].imshow(y, vmin=vmin_top, vmax=vmax_top, origin="lower")
        axesRight[0][2].plot(goals[example_num][0].detach().cpu().numpy(), goals[example_num][1].detach().cpu().numpy(), 'ro', markersize=2)
        axesRight[0][2].set_xticks([], minor=True)
        axesRight[0][2].set_xticks([])
        axesRight[0][2].set_yticks([], minor=True)
        axesRight[0][2].set_yticks([])
        cb_ax = subfigs[1].add_axes([.91,.06,.04,.86])
        fig.colorbar(im, orientation='vertical',cax=cb_ax)

        axesLeft[0].imshow(mask[example_num].detach().cpu().numpy())
        axesLeft[0].set_xticks([], minor=True)
        axesLeft[0].set_xticks([])
        axesLeft[0].set_yticks([], minor=True)
        axesLeft[0].set_yticks([])
        axesLeft[0].invert_yaxis()

        axesLeft[1].imshow(model2(mask)[example_num].detach().cpu().numpy())
        axesLeft[1].set_xticks([], minor=True)
        axesLeft[1].set_xticks([])
        axesLeft[1].set_yticks([], minor=True)
        axesLeft[1].set_yticks([])
        axesLeft[1].invert_yaxis()

        
    subfigs[1].subplots_adjust(left=0, bottom=0.06, right=0.9, top=.93, hspace=.23, wspace=0.1)
    subfigs[1].text(0.32, 0.95, "Ground truth (FMM)", fontsize="medium")
    subfigs[1].text(0.27, 0.48, "Planning neural operator (PNO)", fontsize="medium")
    subfigs[1].text(0.4, 0.02, r"$512\times512$", fontsize="small")
    subfigs[1].text(0.09, 0.02, r"$256\times256$", fontsize="small")
    subfigs[1].text(0.7, 0.02, r"$1024\times1024$", fontsize="small")

    subfigs[0].subplots_adjust(left=0.1, bottom=0.06, right=.9, top=.93, hspace=.23, wspace=0.1)
    subfigs[0].text(0.32, 0.95, r"Input Mask", fontsize="medium")
    subfigs[0].text(0.5, 0.48, "   FNO SDF", fontsize="medium", ha="center", va="center")

    plt.savefig(filename, dpi=300)

batch_size = 2
trainDataCity1024, testDataCity1024 = load_dataset("cityData/1024x1024", trainDataCount=0, testDataCount=1, batch_size=batch_size)
trainDataCity512, testDataCity512 = load_dataset("cityData/512x512", trainDataCount=0, testDataCount=1, batch_size=batch_size)
trainDataCity256, testDataCity256 = load_dataset("cityData/256x256", trainDataCount=0, testDataCount=1, batch_size=batch_size)

plot_paper_fig(testDataCity256, testDataCity512 , testDataCity1024, wrappedPNOwPINN, wrappedSDF, mask_func_arr[3], 0, "superResExampleNYC.pdf")

def plot_example(model, modelPINN, modelSDF, mask_func, data, example_num, filename, normalizer=None):
    set_ticks = True
    height_add = set_size(469, 0.99, (2, 3))[1]
    fig = plt.figure(figsize= set_size(469, 0.99, (2, 3),  height_add=height_add))
    subfigs = fig.subfigures(nrows=1, ncols=1, hspace=0)
    gs1 = subfigs.add_gridspec(2, 6, height_ratios=[1, 1])
    ax1a = subfigs.add_subplot(gs1[0, 0:2])
    ax2a = subfigs.add_subplot(gs1[0, 2:4])
    ax3a = subfigs.add_subplot(gs1[0, 4:6])
    ax4a = subfigs.add_subplot(gs1[1, 0:3])
    ax1 = subfigs.add_subplot(gs1[1, 3:6])

    axes1 = [ax1a, ax2a, ax3a, ax4a, ax1]
    subfigs.suptitle("Value functions and relative error between FMM (Unseen)", fontsize="x-large")

    # Get vmin, vmax
    vmin_top = np.inf
    vmax_top = -np.inf
    vmin_bottom = 0
    vmax_bottom = -np.inf
    loss_func = LpLoss(d=2, p=2)
    set_ticks = False
    vmin_top = 0
    vmax_top = 55
    # PLOT PINN model
    with torch.no_grad():
        for idx, (mask, chi,goals, y) in enumerate(data):
            chi = smooth_chi(mask, modelSDF(mask), 5)    
            out = modelPINN(chi, goals)
            if normalizer is not None:
                out = normalizer.decode(out) * mask_func(mask)
            else:
                out = out*mask    
            if idx == 1:
                break
        axes1[1].imshow(out[example_num].detach().cpu().numpy(), vmin=vmin_top, vmax=vmax_top, origin="lower")
        axes1[1].plot(goals[example_num][0].detach().cpu().numpy(), goals[example_num][1].detach().cpu().numpy(), 'ro')
        axes1[1].contour(out[example_num].reshape(out.shape[1], out.shape[2]).detach().cpu().numpy(),levels=30, colors="k",linewidths=0.3)
        im2 = axes1[4].imshow(torch.abs(out[example_num]-y[example_num]).detach().cpu().numpy(), vmin=0, vmax=5, origin="lower")

        axes1[1].set_title("PNO w/ PINN")
        axes1[4].set_title("PNO w/ \nPINN error")

        if set_ticks:
            axes1[1].set_xticks([0, 250, 500, 750, 1000])
            axes1[1].set_yticks([0, 250, 500, 750, 1000])

    # PLOT without PINN
    with torch.no_grad():
        for idx, (mask, chi,goals, y) in enumerate(data):
            chi = smooth_chi(mask, modelSDF(mask), 5)    
            out = model(chi, goals)
            if normalizer is not None:
                out = normalizer.decode(out) * mask_func(mask)
            else:
                out = out*mask    
            if idx == 1:
                break
    axes1[0].imshow(out[example_num].detach().cpu().numpy(), vmin=vmin_top, vmax=vmax_top, origin="lower")
    axes1[0].plot(goals[example_num][0].detach().cpu().numpy(), goals[example_num][1].detach().cpu().numpy(), 'ro')
    axes1[0].contour(out[example_num].reshape(out.shape[1], out.shape[2]).detach().cpu().numpy(),levels=30, colors="k",linewidths=0.3)
    axes1[0].set_title("PNO w/out PINN")

    # FMM EXAMPLE
    axes1[3].imshow(torch.abs(out[example_num]-y[example_num]).detach().cpu().numpy(), vmin=0, vmax=5, origin="lower")
    axes1[3].set_title("Error PNO \n w/out PINN ")
    if set_ticks:
        axes1[0].set_xticks([0, 250, 500, 750, 1000])
        axes1[0].set_yticks([0, 250, 500, 750, 1000])

    axes1[2].set_xticks([])  
    axes1[2].set_yticks([])    
    axes1[1].set_xticks([])  
    axes1[1].set_yticks([]) 
    axes1[0].set_xticks([])  
    axes1[0].set_yticks([])
    axes1[3].set_xticks([])  
    axes1[3].set_yticks([])
    axes1[4].set_xticks([])  
    axes1[4].set_yticks([])  
        
    y = y[example_num].detach().cpu().numpy()
    im = axes1[2].imshow(y,  cmap='viridis', vmin=vmin_top, vmax=vmax_top, origin="lower")
    axes1[2].plot(goals[example_num][0].detach().cpu().numpy(), goals[example_num][1].detach().cpu().numpy(), 'ro')
    axes1[2].contour(y[:, :, 0],levels=20, colors="k",linewidths=0.3)

    axes1[2].set_title("FMM")
    cb_ax = subfigs.add_axes([.91,.56,.02,.29])
    fig.colorbar(im, orientation='vertical',cax=cb_ax)
    #subfigs.subplots_adjust(left=0.05, bottom=0.03, right=.9, top=0.92, hspace=.32, wspace=0.5)

    cb_ax = subfigs.add_axes([.87,.11,.02,.35])
    fig.colorbar(im2, orientation='vertical',cax=cb_ax)
    plt.savefig(filename, dpi=300)
    plt.show()
    #subfigs.subplots_adjust(left=0.05, bottom=0.1, right=.9, top=0.85, hspace=.4, wspace=0.3)

def plot_example_grad(model, modelPINN, modelSDF, mask_func, data, example_num, filename, normalizer=None):
    set_ticks = True
    height_add = set_size(469, 0.99, (2, 3))[1]
    fig = plt.figure(figsize= set_size(469, 0.99, (2, 3),  height_add=height_add))
    subfigs = fig.subfigures(nrows=1, ncols=1, hspace=0)
    gs1 = subfigs.add_gridspec(2, 6)
    ax1a = subfigs.add_subplot(gs1[0, 0:2])
    ax2a = subfigs.add_subplot(gs1[0, 2:4])
    ax5a = subfigs.add_subplot(gs1[0, 4:6])

    ax3a = subfigs.add_subplot(gs1[1, 0:3])
    ax4a = subfigs.add_subplot(gs1[1, 3:6])

    axes1 = [ax1a, ax2a, ax3a, ax4a, ax5a]
    subfigs.suptitle(r"Gradient of value function ($\|\nabla V(x)\|$) and error \\ of gradient value function ($|\|\nabla V(x)\|-c(x)|$) (Seen)", fontsize="x-large")

    # Get vmin, vmax
    vmin_top = np.inf
    vmax_top = -np.inf
    vmin_bottom = 0
    vmax_bottom = -np.inf
    loss_func = LpLoss(d=2, p=2)
    set_ticks = False
    vmin_top = 0
    vmax_top = 2
    # PLOT PINN MODEL
    with torch.no_grad():
        for idx, (mask, chi,goals, y) in enumerate(data):
            chi = smooth_chi(mask, modelSDF(mask), 5)    
            out = modelPINN(chi, goals)
            if normalizer is not None:
                out = normalizer.decode(out) * mask_func(mask)
            else:
                out = out*mask    
            if idx == 1:
                break
    grad_out = torch.linalg.vector_norm(torch.stack(list(torch.gradient(out[example_num].reshape(1, out.shape[1], out.shape[2]), dim=[1, 2])), dim=0), dim=0)
    grad_y = torch.linalg.vector_norm(torch.stack(list(torch.gradient(y[example_num].reshape(1, y.shape[1], y.shape[2]), dim=[1, 2])), dim=0), dim=0)
    im= axes1[1].imshow(grad_out[0].detach().cpu().numpy(), vmin=vmin_top, vmax=vmax_top, origin="lower")
    axes1[1].plot(goals[example_num][0].detach().cpu().numpy(), goals[example_num][1].detach().cpu().numpy(), 'ro')

    axes1[1].set_title("PNO w/ PINN")
    
    if set_ticks:
        axes1[1].set_xticks([0, 250, 500, 750, 1000])
        axes1[1].set_yticks([0, 250, 500, 750, 1000])

    imOMEGA = axes1[3].imshow(torch.abs(grad_out[0]-grad_y[0]).detach().cpu().numpy(), vmin=0, vmax=1.5, origin="lower")
    axes1[3].set_title("PNO w PINN\n Gradient Error")
   
    with torch.no_grad():
        for idx, (mask, chi,goals, y) in enumerate(data):
            chi = smooth_chi(mask, modelSDF(mask), 5)    
            out = model(chi, goals)
            if normalizer is not None:
                out = normalizer.decode(out) * mask_func(mask)
            else:
                out = out*mask    
            if idx == 1:
                break

    grad_out = torch.linalg.vector_norm(torch.stack(list(torch.gradient(out[example_num].reshape(1, out.shape[1], out.shape[2]), dim=[1, 2])), dim=0), dim=0)
    grad_y = torch.linalg.vector_norm(torch.stack(list(torch.gradient(y[example_num].reshape(1, y.shape[1], y.shape[2]), dim=[1, 2])), dim=0), dim=0)
    
    axes1[4].imshow(grad_y[0].detach().cpu().numpy(), vmin=vmin_top, vmax=vmax_top, origin="lower")
    axes1[4].plot(goals[example_num][0].detach().cpu().numpy(), goals[example_num][1].detach().cpu().numpy(), 'ro')
    axes1[4].set_title("FMM")
    axes1[0].set_title("PNO w/out PINN")
    print(loss_func(grad_out.view(1, y.shape[1], y.shape[2]), grad_y.view(1, y.shape[1], y.shape[2])))
    axes1[0].imshow(grad_out[0].detach().cpu().numpy(), vmin=vmin_top, vmax=vmax_top, origin="lower")
    axes1[0].plot(goals[example_num][0].detach().cpu().numpy(), goals[example_num][1].detach().cpu().numpy(), 'ro')
    if set_ticks:
        axes1[0].set_xticks([0, 250, 500, 750, 1000])
        axes1[0].set_yticks([0, 250, 500, 750, 1000])
    axes1[2].imshow(torch.abs(grad_out[0]-grad_y[0]).detach().cpu().numpy(), vmin=0, vmax=1.5, origin="lower")
    axes1[2].set_title("PNO w/out PINN\n Gradient Error")
    axes1[2].set_xticks([])  
    axes1[2].set_yticks([])    
    axes1[1].set_xticks([])  
    axes1[1].set_yticks([]) 
    axes1[0].set_xticks([])  
    axes1[0].set_yticks([])
    axes1[3].set_xticks([])  
    axes1[3].set_yticks([])  
    axes1[4].set_xticks([])  
    axes1[4].set_yticks([])

    cb_ax = subfigs.add_axes([.91,.54,.02,.33])
    fig.colorbar(im, orientation='vertical',cax=cb_ax)
    #subfigs.subplots_adjust(left=0.05, bottom=0.0, right=.9, top=1, hspace=-0.3, wspace=0.5)

    cb_ax = subfigs.add_axes([.91,.13,.02,.33])
    fig.colorbar(imOMEGA, orientation='vertical',cax=cb_ax)
    plt.savefig(filename, dpi=300)
    plt.show()
    #subfigs.subplots_adjust(left=0.05, bottom=0.1, right=.9, top=0.85, hspace=.4, wspace=0.3)

_, testData64 = load_dataset("synthetic/64x64", 0.8, 0.2, batch_size=20)
plot_example(wrappedPNO, wrappedPNOwPINN, wrappedSDF, mask_func_arr[2], testData64, 2, "pinnOUTDIST2.pdf")
plot_example_grad(wrappedPNO, wrappedPNOwPINN, wrappedSDF, mask_func_arr[2], testData64, 2, "pinnGradOUTDIST2.pdf")
