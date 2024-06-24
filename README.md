# Finetuning-LLM-using-LoRA-and-QLoRA

There are so many computational challenges when fine-tuning the LLM. We all know larger the number of nuerons or parameter in LLM, better will be its intelligence.

For storing one parameter of 32 bit floating point precision we need 4 bytes memory.

| Storage | Memory Needed | 
|---------|-------------|
| Model Parameter | 4 bytes per parameter  | 

Similarly, we need storage space for storing the training parameters.

| Training | Memory Needed | 
|---------|-------------|
| Adam optimizer | 8 bytes per parameter  | 
| Gradient | 4 bytes per parameter | 
| Activation | 8 bytes per parameter | 

So, we can say that for storing and training single parameter we need 24 bytes. If the model have 1 billion parameter then we need around 24x10^9 bytes that is 24 GB. Nowadays, the model size
is more than 10-12 billion parameters. Hence it will be computationally very difficult to train (requires very huge GPU) the entrie model on our dataset.

## How can we overcome this problem ?
This problem can be solved by intoducing the quantization. In a simple way, In quantization, the memory needed for storing the model parameters are reduced by reducing its precision from 32 
floating point to 16 bit floating point or 8 bit integer precision. Also, We can implement paramater efficient fine tuning.

### Parameter efficient fine tuning (PEFT)
We can broadly divide the PEFT method into 3 classes and they are:
1. Selective
2. Reparameterization (LoRA)
3. Additive

#### Selective Class
This method will finetune only a subset of LLM parameter. There are several ways in which we can identify which parameter needs to updated or selected for finetuning process. 
The result obtained from this class is mix. Sometime it will generate great result and sometimes not.

#### Reparameterization
It will create a new low rank representation of the original network range. This is how LoRA or reparameterization reduces the number of parameters needed for the model training.
The results were great in almost all tasks.

#### Additive 
It will keeps all the LLM weights frozen and it will introduce or add new trainable components or layer to the LLM. There are two main approaches and they are:
1. Adapters
2. Soft prompts

Adapater method will add a new trainable layer to the exsiting acrchitecture of the model. It will be typically inside the encoding or decoding layers, after the attention of fixed
forward layer. However In the Soft Prompts method the model architecture will be fixed. They will try to manipulate the data to imporve the model perfromance. This can be achieved
by adding the trainable parameters to the prompt embeddings or keeping the input fixed and re-training the embedding weights.





### References

Generative AI with Large Langugage Models - Coursera
Low Rank Adaptation (LoRA): From Intution to Implementation by Harsh Maheshwari (Medium Article)


