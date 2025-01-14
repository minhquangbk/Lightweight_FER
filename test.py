from code_base.models.hub.fgw import Model

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_layers(model, level=0):
    # Check if this is the top-level call and not a recursive one
    # if level == 0:
    #     print(model.__class__.__name__)

    # Iterate through the children of the model/module
    for name, child in model.named_children():
        # Indentation to reflect hierarchy
        indent = '\t' * level

        # Print the name and class of the child
        print(f'{indent} {name}: False')

        # Recursively call this function for the child, if it has further children
        if list(child.children()):
            print_model_layers(child, level+1)

model = Model()
print_model_layers(model)
# print(model)
# exit()
# print(count_parameters(model))
# exit()
# for name, param in model.named_parameters():
#     # print(param)
#     print(f"Layer: {name}")
#     print(f" - Weight: {param.shape}")  # This prints the shape of the weights
#     print('===============================')
#     print('===============================')
