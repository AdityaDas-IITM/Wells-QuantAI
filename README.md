# Wells-QuantAI
## Runners-Up out of 586 Participating teams
### Team Third Degree Burn's Solution to the [Wells Fargo Quant AI Hack](https://dare2compete.com/hackathon/wells-fargo-quantitative-ai-hackathon-shaastra-2022-indian-institute-of-technology-iit-madras-248431)


## Solution Outline

### Dataset Exploration
- The data is clean, there are no NaN values and no datatype conflicts
- Every weekend in the dataset has been skipped and there are no other holidays
- The Volatility values lie between between 0 and 1
- We also noticed some patterns noticed between moneynesses for a particular tenor

### Correlation Between Different Moneyness

As you can see from the image below we observe that there is a correlation between the volatility values for different moneyness values. One kind of trend is followed for all Moneyness values less than 1 and a different one is followed for all moneyness values greater than 1.
![](Assets/Moneyness_2M.png)

### Model Architecture
#### Encoder: 
- Multiple blocks of multi-headed self attention and a layer normalisation with residual connections. 
- Input: Flattened IV values with the positional embeddings (will be explained later)
#### Decoder: 
- GRU layer: predicts some time steps into the future. 
- Input: Previous output of the GRU and a context vector (weighted average of encoder outputs). 
- Weights calculated conditional on current hidden state of the GRU and the encoder outputs. This setup allows us to have a large input sequence sizes.

Below shown is an image of what the architecture looks like:
<p align="center">
  <img src="https://github.com/AdityaDas-IITM/Wells-QuantAI/blob/main/Assets/Model.png" />
</p>

### Creating Positional Embeddings for the Dates
- Positional embedding of an element is proportional to difference between its date and the date of the first element in the input sequence. 
- This Accounts for any turbulence on weekends and other market-closed days leading to abrupt changes in price.
![](Assets/Positional_Embeddings.png)

### Our Loss Function
The Loss function is a combination of RMSE, temporal continuity and spatial loss. (All of which we explain subsequently).
![](Assets/Loss_fns.png)
