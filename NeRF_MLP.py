import torch
import torch.nn as nn


class NeRF_MLP(nn.Module):

    def __init__(self, L_pos=10, L_dir=4, hidden_dim=256):
        super(NeRF_MLP, self).__init__()

        # Frequency of encoding
        self.L_pos = L_pos
        self.L_dir = L_dir

        # Fully connected layers
        # Block 1:
        self.fc1 = nn.Linear(L_pos * 6 + 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        # Block 2:
        self.fc6 = nn.Linear(hidden_dim + L_pos * 6 + 3, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, hidden_dim)
        self.fc9 = nn.Linear(hidden_dim, hidden_dim + 1)
        # Block 3:
        self.fc10 = nn.Linear(hidden_dim + L_dir * 6 + 3, hidden_dim // 2)
        self.fc11 = nn.Linear(hidden_dim // 2, 3)

        # Non-linearities
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def positional_encoding(slf, x: torch.Tensor, L: int) -> torch.Tensor:
        """
        Apply positional encoding to the input coordinates (x, y, z).

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3) representing N sets of 3D coordinates.
            L (int): Parameter for controlling the frequency of encoding.

        Returns:
            torch.Tensor: Positionally encoded features for the input coordinates of shape (N, 3 + 6 * L).
        """
        encoding_components = []

        # Loop over encoding frequencies up to L
        for j in range(L):
            # Calculate sine and cosine components for each frequency
            encoding_components.append(torch.sin(2 ** j * x))
            encoding_components.append(torch.cos(2 ** j * x))

        # Concatenate the original input with the encoding components
        encoded_coordinates = torch.cat([x] + encoding_components, dim=1)

        return encoded_coordinates

    def forward(self, xyz, d):
        x_emb = self.positional_encoding(xyz, self.L_pos)  # [batch_size, Lpos * 6 + 3]
        print("Shape of x_emb: ", x_emb.shape)
        d_emb = self.positional_encoding(d, self.L_dir)  # [batch_size, Ldir * 6 + 3]
        print("Shape of d_emb: ", d_emb.shape)

        ### ------------ Block 1:
        x = self.fc1(x_emb)  # [batch_size, hidden_dim]
        x = self.relu(x)
        print("Shape after fc1:", x.shape)

        x = self.fc2(x)
        x = self.relu(x)
        print("Shape after fc2:", x.shape)

        x = self.fc3(x)
        x = self.relu(x)
        print("Shape after fc3:", x.shape)

        x = self.fc4(x)
        x = self.relu(x)
        print("Shape after fc4:", x.shape)

        x = self.fc5(x)
        x = self.relu(x)
        print("Shape after fc5:", x.shape)

        ### ------------ Block 2:
        x = self.fc6(torch.cat((x, x_emb), dim=1)) #skip connection
        x = self.relu(x)
        print("Shape after fc6:", x.shape)

        x = self.fc7(x)
        x = self.relu(x)
        print("Shape after fc7:", x.shape)

        x = self.fc8(x)
        x = self.relu(x)
        print("Shape after fc8:", x.shape)

        x = self.fc9(x)
        print("Shape after fc9:", x.shape)

        ### ------------ Block 3:

        # Extract sigma from x (last value)
        sigma = x[:, -1]

        # Density
        density = torch.relu(sigma) #torch.Size([16])
        print("Shape of density:", density.shape)

        # Take all values from except sigma (everything except last one)
        x = x[:, :-1]  # [batch_size, hidden_dim] #torch.Size([16, 256])
        print("Shape of x only:", x.shape)

        x = self.fc10(torch.cat((x, d_emb), dim=1))
        x = self.relu(x)
        print("Shape after fc10:", x.shape)

        color = self.fc11(x)
        color = self.sigmoid(color)
        print("Shape after fc11:", color.shape)

        return color, density

    # def intersect(self, x, d):
    #     return self.forward(x, d)


if __name__ == "__main__":
    # Define hyperparameters
    L_pos = 10
    L_dir = 4
    hidden_dim = 256
    batch_size = 16

    # Create instance of Nerf model
    model = NeRF_MLP(L_pos, L_dir, hidden_dim)

    # Simulated data
    xyz = torch.randn(batch_size, 3)
    print(xyz.shape)
    d = torch.randn(batch_size, 3)
    print(d.shape)

    # Forward pass
    color, density = model.forward(xyz, d)

    # Print shapes of outputs
    print("Density shape:", density.shape)
    print("Color shape:", color.shape)
