import matplotlib.pyplot as plt
import re

def extract_training_info(training_output):
    # Define regex patterns for losses and accuracies
    pattern_loss = r"Train Loss: ([\d.]+).*Test Loss: ([\d.]+)"
    pattern_acc = r"Train Acc: ([\d.]+).*Test Acc: ([\d.]+)"
    
    # Initialize lists to hold the extracted values
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    
    # Split the output into lines for processing
    lines = training_output.strip().split('\n')
    
    # Process each line
    for line in lines:
        # Search for loss values
        loss_match = re.search(pattern_loss, line)
        if loss_match:
            train_loss, test_loss = loss_match.groups()
            train_losses.append(float(train_loss))
            test_losses.append(float(test_loss))
        
        # Search for accuracy values
        acc_match = re.search(pattern_acc, line)
        if acc_match:
            train_acc, test_acc = acc_match.groups()
            train_accs.append(float(train_acc))
            test_accs.append(float(test_acc))
    
    # Return the extracted information
    return train_losses, test_losses, train_accs, test_accs


training_output = """
testing...........
*====**
0m 11s
Epoch: 000, Train Loss: 1.3008940, Train Acc: 0.4637681, Test Loss: 1.1134306, Test Acc: 0.5120773
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 001, Train Loss: 1.3241710, Train Acc: 0.5314010, Test Loss: 1.1266221, Test Acc: 0.4927536
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 002, Train Loss: 1.2347655, Train Acc: 0.5297907, Test Loss: 1.3039593, Test Acc: 0.5072464
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 003, Train Loss: 1.1661876, Train Acc: 0.5265700, Test Loss: 1.6773293, Test Acc: 0.4782609
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 004, Train Loss: 1.2893730, Train Acc: 0.5346216, Test Loss: 1.3725259, Test Acc: 0.5024155
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 005, Train Loss: 1.3401730, Train Acc: 0.5217391, Test Loss: 1.0782597, Test Acc: 0.4492754
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 006, Train Loss: 1.2227930, Train Acc: 0.4750403, Test Loss: 1.0156603, Test Acc: 0.5314010
saving best model
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 007, Train Loss: 1.1314876, Train Acc: 0.5152979, Test Loss: 1.1502646, Test Acc: 0.4685990
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 008, Train Loss: 1.1042614, Train Acc: 0.4879227, Test Loss: 1.0077872, Test Acc: 0.4879227
saving best model
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 009, Train Loss: 1.0875399, Train Acc: 0.4718196, Test Loss: 1.0027392, Test Acc: 0.5120773
saving best model
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 010, Train Loss: 1.0741712, Train Acc: 0.4669887, Test Loss: 1.0324020, Test Acc: 0.5072464
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 011, Train Loss: 1.0493951, Train Acc: 0.4573269, Test Loss: 0.9998878, Test Acc: 0.5120773
saving best model
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 012, Train Loss: 1.0379746, Train Acc: 0.5265700, Test Loss: 0.9949506, Test Acc: 0.5024155
saving best model
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 013, Train Loss: 1.0373975, Train Acc: 0.5281804, Test Loss: 0.9902998, Test Acc: 0.4879227
saving best model
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 014, Train Loss: 0.9881724, Train Acc: 0.5056361, Test Loss: 0.9941536, Test Acc: 0.4975845
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 015, Train Loss: 0.9969055, Train Acc: 0.4669887, Test Loss: 1.0040373, Test Acc: 0.5072464
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 016, Train Loss: 1.0110279, Train Acc: 0.5652174, Test Loss: 1.0484013, Test Acc: 0.5072464
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 017, Train Loss: 1.0197169, Train Acc: 0.5780998, Test Loss: 0.9929936, Test Acc: 0.5169082
train...........
LR 0.01
testing...........
*====**
0m 10s
Epoch: 018, Train Loss: 0.9771562, Train Acc: 0.4685990, Test Loss: 1.0434304, Test Acc: 0.5024155
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 019, Train Loss: 0.9448987, Train Acc: 0.4669887, Test Loss: 1.1608240, Test Acc: 0.5072464
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 020, Train Loss: 0.9528275, Train Acc: 0.4685990, Test Loss: 1.1798086, Test Acc: 0.5072464
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 021, Train Loss: 0.9440505, Train Acc: 0.5925926, Test Loss: 1.0655612, Test Acc: 0.5217391
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 022, Train Loss: 0.9320095, Train Acc: 0.5330113, Test Loss: 1.1534196, Test Acc: 0.4927536
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 023, Train Loss: 0.8941716, Train Acc: 0.5378422, Test Loss: 1.0504984, Test Acc: 0.4879227
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 024, Train Loss: 0.9058135, Train Acc: 0.5362319, Test Loss: 2.3411668, Test Acc: 0.4830918
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 025, Train Loss: 0.9098591, Train Acc: 0.5008052, Test Loss: 0.9739224, Test Acc: 0.5265700
saving best model
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 026, Train Loss: 0.9164594, Train Acc: 0.6328502, Test Loss: 0.9497304, Test Acc: 0.5748792
saving best model
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 027, Train Loss: 0.8911465, Train Acc: 0.6231884, Test Loss: 0.9895394, Test Acc: 0.5314010
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 028, Train Loss: 0.8891115, Train Acc: 0.4669887, Test Loss: 1.1130529, Test Acc: 0.5072464
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 029, Train Loss: 0.8770219, Train Acc: 0.6022544, Test Loss: 1.2238978, Test Acc: 0.4975845
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 030, Train Loss: 0.8352631, Train Acc: 0.4669887, Test Loss: 1.1843002, Test Acc: 0.5072464
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 031, Train Loss: 0.8698712, Train Acc: 0.5636071, Test Loss: 1.4800325, Test Acc: 0.5024155
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 032, Train Loss: 0.8474466, Train Acc: 0.4669887, Test Loss: 1.0679832, Test Acc: 0.5072464
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 033, Train Loss: 0.8286434, Train Acc: 0.5813205, Test Loss: 1.8212551, Test Acc: 0.4975845
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 034, Train Loss: 0.8236436, Train Acc: 0.7053140, Test Loss: 0.9725238, Test Acc: 0.5700483
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 035, Train Loss: 0.8180246, Train Acc: 0.4975845, Test Loss: 1.2538124, Test Acc: 0.5217391
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 036, Train Loss: 0.8148665, Train Acc: 0.5394525, Test Loss: 2.5253074, Test Acc: 0.4927536
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 037, Train Loss: 0.8133208, Train Acc: 0.4669887, Test Loss: 1.3711404, Test Acc: 0.5072464
train...........
LR 0.005
testing...........
*====**
0m 10s
Epoch: 038, Train Loss: 0.8745056, Train Acc: 0.6972625, Test Loss: 0.9894774, Test Acc: 0.5942029
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 039, Train Loss: 0.8051065, Train Acc: 0.7665056, Test Loss: 1.0174068, Test Acc: 0.5748792
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 040, Train Loss: 0.7768042, Train Acc: 0.6505636, Test Loss: 1.0023253, Test Acc: 0.5797101
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 041, Train Loss: 0.7708505, Train Acc: 0.7842190, Test Loss: 0.9909771, Test Acc: 0.5893720
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 042, Train Loss: 0.7593872, Train Acc: 0.6022544, Test Loss: 1.1628804, Test Acc: 0.5458937
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 043, Train Loss: 0.7471886, Train Acc: 0.6505636, Test Loss: 1.4363734, Test Acc: 0.5652174
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 044, Train Loss: 0.7683192, Train Acc: 0.5974235, Test Loss: 1.1552729, Test Acc: 0.5458937
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 045, Train Loss: 0.7481704, Train Acc: 0.7117552, Test Loss: 1.2043108, Test Acc: 0.5700483
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 046, Train Loss: 0.7528981, Train Acc: 0.7987118, Test Loss: 1.0739115, Test Acc: 0.5893720
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 047, Train Loss: 0.7402973, Train Acc: 0.6795491, Test Loss: 1.2979630, Test Acc: 0.5555556
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 048, Train Loss: 0.7555224, Train Acc: 0.5169082, Test Loss: 1.1708477, Test Acc: 0.5217391
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 049, Train Loss: 0.7418712, Train Acc: 0.8051530, Test Loss: 1.1297704, Test Acc: 0.5748792
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 050, Train Loss: 0.7281891, Train Acc: 0.8115942, Test Loss: 1.1176464, Test Acc: 0.5893720
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 051, Train Loss: 0.7191699, Train Acc: 0.7842190, Test Loss: 1.0204296, Test Acc: 0.5797101
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 052, Train Loss: 0.7113312, Train Acc: 0.6795491, Test Loss: 1.4309778, Test Acc: 0.5507246
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 053, Train Loss: 0.7016708, Train Acc: 0.4863124, Test Loss: 1.4595517, Test Acc: 0.4879227
train...........
LR 0.0025
testing...........
*====**
0m 11s
Epoch: 054, Train Loss: 0.7484636, Train Acc: 0.5410628, Test Loss: 5.2189510, Test Acc: 0.4927536
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 055, Train Loss: 0.7458676, Train Acc: 0.5426731, Test Loss: 1.0883406, Test Acc: 0.4541063
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 056, Train Loss: 0.7588965, Train Acc: 0.5636071, Test Loss: 1.0775232, Test Acc: 0.5265700
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 057, Train Loss: 0.7235313, Train Acc: 0.7648953, Test Loss: 1.0548288, Test Acc: 0.5652174
train...........
LR 0.0025
testing...........
*====**
0m 10s
Epoch: 058, Train Loss: 0.7142651, Train Acc: 0.6521739, Test Loss: 1.1278400, Test Acc: 0.5603865
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 059, Train Loss: 0.6961467, Train Acc: 0.7729469, Test Loss: 1.3208220, Test Acc: 0.5555556
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 060, Train Loss: 0.6967541, Train Acc: 0.8164251, Test Loss: 1.3090539, Test Acc: 0.5507246
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 061, Train Loss: 0.7058197, Train Acc: 0.6135266, Test Loss: 2.2578570, Test Acc: 0.5169082
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 062, Train Loss: 0.6940631, Train Acc: 0.6425121, Test Loss: 1.4338627, Test Acc: 0.5362319
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 063, Train Loss: 0.6790215, Train Acc: 0.8180354, Test Loss: 1.0984124, Test Acc: 0.5555556
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 064, Train Loss: 0.6773114, Train Acc: 0.8373591, Test Loss: 1.2157459, Test Acc: 0.5652174
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 065, Train Loss: 0.6697846, Train Acc: 0.6908213, Test Loss: 1.3557586, Test Acc: 0.5314010
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 066, Train Loss: 0.6674429, Train Acc: 0.7326892, Test Loss: 1.1063099, Test Acc: 0.5845411
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 067, Train Loss: 0.6479951, Train Acc: 0.6811594, Test Loss: 1.6381136, Test Acc: 0.5314010
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 068, Train Loss: 0.6465205, Train Acc: 0.6247987, Test Loss: 2.0991697, Test Acc: 0.5169082
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 069, Train Loss: 0.6676709, Train Acc: 0.8228663, Test Loss: 1.4552979, Test Acc: 0.5555556
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 070, Train Loss: 0.6479789, Train Acc: 0.8631240, Test Loss: 1.3066362, Test Acc: 0.5700483
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 071, Train Loss: 0.6335399, Train Acc: 0.8647343, Test Loss: 1.2498102, Test Acc: 0.5748792
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 072, Train Loss: 0.6377329, Train Acc: 0.6038647, Test Loss: 1.4586913, Test Acc: 0.5024155
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 073, Train Loss: 0.6696717, Train Acc: 0.8389694, Test Loss: 1.2987407, Test Acc: 0.5603865
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 074, Train Loss: 0.6219503, Train Acc: 0.8550725, Test Loss: 1.3565635, Test Acc: 0.5797101
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 075, Train Loss: 0.6320597, Train Acc: 0.7858293, Test Loss: 1.1491428, Test Acc: 0.5507246
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 076, Train Loss: 0.6226018, Train Acc: 0.8937198, Test Loss: 1.3138348, Test Acc: 0.5652174
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 077, Train Loss: 0.6083875, Train Acc: 0.7552335, Test Loss: 1.4736076, Test Acc: 0.5410628
train...........
LR 0.00125
testing...........
*====**
0m 10s
Epoch: 078, Train Loss: 0.6260581, Train Acc: 0.7246377, Test Loss: 1.1236784, Test Acc: 0.5265700
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 079, Train Loss: 0.6124382, Train Acc: 0.8647343, Test Loss: 1.1832238, Test Acc: 0.5265700
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 080, Train Loss: 0.6414640, Train Acc: 0.7407407, Test Loss: 1.7814634, Test Acc: 0.5507246
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 081, Train Loss: 0.6154629, Train Acc: 0.8003221, Test Loss: 1.6934519, Test Acc: 0.5652174
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 082, Train Loss: 0.6191417, Train Acc: 0.8486312, Test Loss: 1.2534454, Test Acc: 0.5314010
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 083, Train Loss: 0.5987373, Train Acc: 0.7037037, Test Loss: 1.1542177, Test Acc: 0.5120773
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 084, Train Loss: 0.6086705, Train Acc: 0.8293076, Test Loss: 1.3402919, Test Acc: 0.5265700
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 085, Train Loss: 0.5854682, Train Acc: 0.8470209, Test Loss: 1.6326067, Test Acc: 0.5797101
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 086, Train Loss: 0.6097935, Train Acc: 0.8067633, Test Loss: 1.6356641, Test Acc: 0.5748792
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 087, Train Loss: 0.5871941, Train Acc: 0.8599034, Test Loss: 1.2572922, Test Acc: 0.5748792
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 088, Train Loss: 0.5836500, Train Acc: 0.9049919, Test Loss: 1.3190999, Test Acc: 0.5845411
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 089, Train Loss: 0.5739327, Train Acc: 0.9001610, Test Loss: 1.4736722, Test Acc: 0.5458937
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 090, Train Loss: 0.5813410, Train Acc: 0.8582931, Test Loss: 1.6180766, Test Acc: 0.5603865
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 091, Train Loss: 0.5991053, Train Acc: 0.9033816, Test Loss: 1.4472117, Test Acc: 0.5603865
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 092, Train Loss: 0.5865749, Train Acc: 0.8083736, Test Loss: 1.7053249, Test Acc: 0.5652174
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 093, Train Loss: 0.5796507, Train Acc: 0.8969404, Test Loss: 1.4614289, Test Acc: 0.5458937
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 094, Train Loss: 0.5753944, Train Acc: 0.8888889, Test Loss: 1.4959026, Test Acc: 0.5603865
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 095, Train Loss: 0.5655090, Train Acc: 0.8276973, Test Loss: 1.2818155, Test Acc: 0.5314010
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 096, Train Loss: 0.5588648, Train Acc: 0.8357488, Test Loss: 1.4650812, Test Acc: 0.5652174
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 097, Train Loss: 0.5850131, Train Acc: 0.6553945, Test Loss: 1.6117863, Test Acc: 0.5169082
train...........
LR 0.000625
testing...........
*====**
0m 10s
Epoch: 098, Train Loss: 0.5710100, Train Acc: 0.9194847, Test Loss: 1.6013374, Test Acc: 0.5603865
train...........
LR 0.0003125
testing...........
*====**
0m 10s
Epoch: 099, Train Loss: 0.5679940, Train Acc: 0.8051530, Test Loss: 1.6895247, Test Acc: 0.5410628
testing...........
"""


# Extract the information from the training output (assuming the function extract_training_info is defined)
train_loss, test_loss, train_acc, test_acc = extract_training_info(training_output)
epochs = list(range(1, len(train_loss) + 1))

# Create two separate plots for training and testing metrics
fig, axs = plt.subplots(2, figsize=(10, 10))

# Plotting train loss and accuracy
axs[0].plot(epochs, train_loss, label='Train Loss', color='red', marker='o')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Train Loss')
axs[0].set_title('Training Loss Over Epochs')
axs[0].legend(loc='upper left')

# Adding secondary axis for train accuracy percentage
ax_acc = axs[0].twinx()
ax_acc.plot(epochs, train_acc, label='Train Accuracy', color='blue', marker='o')
ax_acc.set_ylabel('Train Accuracy (%)')
ax_acc.legend(loc='upper right')

# Plotting test loss and accuracy
axs[1].plot(epochs, test_loss, label='Test Loss', color='red', marker='x')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Test Loss')
axs[1].set_title('Testing Loss Over Epochs')
axs[1].legend(loc='upper left')

# Adding secondary axis for test accuracy percentage
ax_t_acc = axs[1].twinx()
ax_t_acc.plot(epochs, test_acc, label='Test Accuracy', color='blue', marker='x')
ax_t_acc.set_ylabel('Test Accuracy (%)')
ax_t_acc.legend(loc='upper right')

plt.tight_layout()
plt.show()

# epochs = list(range(1, len(train_loss) + 1))

# # Create separate figures for train and test metrics
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# # Plotting train loss and accuracy
# ax1.plot(epochs, train_loss, label='Train Loss', color='red', marker='o')
# ax1.plot(epochs, train_acc, label='Train Accuracy', color='blue', marker='o')
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Train Loss/Accuracy')
# ax1.set_title('Training Loss and Accuracy')
# ax1.legend()

# # Plotting test loss and accuracy
# ax2.plot(epochs, test_loss, label='Test Loss', color='darkred', marker='x')
# ax2.plot(epochs, test_acc, label='Test Accuracy', color='darkblue', marker='x')
# ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Test Loss/Accuracy')
# ax2.set_title('Testing Loss and Accuracy')
# ax2.legend()

# plt.tight_layout()
# plt.show()


# train_loss, test_loss, train_acc, test_acc = extract_training_info(training_output)
# epochs = list(range(len(train_loss)))
# #
# # Create a figure and a set of subplots
# fig, ax1 = plt.subplots()

# # Plotting train and test accuracy on the same graph but different axes
# color = 'tab:red'
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Train/Test Loss', color=color)
# ax1.plot(epochs, train_loss, label='Train Loss', color='red', marker='o')
# ax1.plot(epochs, test_loss, label='Test Loss', color='darkred', marker='x')
# ax1.tick_params(axis='y', labelcolor=color)

# # Instantiate a second axes that shares the same x-axis
# ax2 = ax1.twinx()  
# color = 'tab:blue'
# ax2.set_ylabel('Train/Test Accuracy', color=color)  # we already handled the x-label with ax1
# ax2.plot(epochs, train_acc, label='Train Accuracy', color='blue', marker='o')
# ax2.plot(epochs, test_acc, label='Test Accuracy', color='darkblue', marker='x')
# ax2.tick_params(axis='y', labelcolor=color)

# # Adding legends
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.title('Development of Train/Test Loss and Accuracy')
# plt.show()

