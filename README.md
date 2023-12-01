# STERN: Attention-driven Spatial Transformer Network for Abnormality Detection in Chest X-Ray Images

### Abstract

### How to Use

#### Requirements
This code was developed using a Pytorch framework. The file with all the requirements is included in the repository (*requirements_env.yml*).

#### File Structure
*models.py* - Model classes of the spatial transformer network implemented for attention purposes.

*loss.py* - Loss function developed for finetuning, using a scaling factor loss term.

#### Example
Import scripts and initialize the model.
```ruby
from models import *
from loss import *

model = STERN_attention_network(scaling_factor=1.25)
```

Forward and backward steps (first training stage).
```ruby
...
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs))
    model.train()
    for batch_idx, (data, target, path) in enumerate(train_loader):  
        
        data, target = data.to(device), target.to(device)
        output = model(data)
        output = torch.squeeze(output,axis=1)

        loss = STERN_loss(output, target)
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
...
```

Forward and backward steps (finetuning training stage, with a scaling factor loss term).
```ruby
...
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs))
    model.train()
    for batch_idx, (data, target, path) in enumerate(train_loader):  
        
        data, target = data.to(device), target.to(device)
        output = model(data)
        output = torch.squeeze(output,axis=1)
        
        scaling_factor_term = torch.squeeze(model.stn(data)[2])-model.scaling_factor  #sx/sy-Sf

        loss = STERN_loss(output, target,scaling_factor_term=scaling_factor_term)
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
...
```

### Credits
If you use this code, please cite the following publications: 

**Rocha, Joana, et al. "STERN: Attention-driven Spatial Transformer Network for abnormality detection in chest X-ray images." Artificial intelligence in medicine 147 (2024): 102737. [https://doi.org/10.1109/CBMS55023.2022.00051](https://doi.org/10.1016/j.artmed.2023.102737)**

**Rocha, Joana, et al. "Attention-driven Spatial Transformer Network for Abnormality Detection in Chest X-Ray Images." 2022 IEEE 35th International Symposium on Computer-Based Medical Systems (CBMS). IEEE, 2022. [https://doi.org/10.1109/CBMS55023.2022.00051](https://doi.org/10.1109/CBMS55023.2022.00051)**



