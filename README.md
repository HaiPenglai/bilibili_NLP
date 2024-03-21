当前进度：还在更新

文档：数学公式，手绘图，代码，注释，文字
手绘板+代码

### 基础语法

#### 创建没有初始化的矩阵

在PyTorch中，你可以使用`torch.empty`函数创建一个没有初始化的矩阵。以下是一个简单的例子：

```python
import torch

# 创建一个大小为3x3的未初始化矩阵
uninitialized_matrix = torch.empty(3, 3)

# 输出未初始化矩阵
print("未初始化矩阵:")
print(uninitialized_matrix)
```

这里，`torch.empty`函数创建了一个指定大小的未初始化矩阵。请注意，这个矩阵的值将取决于内存中的随机值，因此输出的矩阵元素可能是任意值。

如果你需要创建一个特定值的矩阵，你可以使用其他初始化函数，比如`torch.zeros`（创建全零矩阵）或`torch.ones`（创建全一矩阵）。



#### 创建随机初始化的矩阵

当然，你可以使用`torch.rand`函数创建一个在0到1之间均匀分布的随机矩阵。以下是在刚才的代码后面演示如何创建一个有初始化的随机矩阵并输出：

```python
import torch

# 创建一个大小为3x3的随机矩阵（均匀分布）
random_matrix = torch.rand(3, 3)

# 输出随机矩阵
print("随机矩阵:")
print(random_matrix)
```

在这个例子中，`torch.rand`函数创建了一个大小为3x3的矩阵，其中的元素是从0到1的均匀分布中随机抽取的。输出的矩阵将包含在[0, 1)范围内的随机数。
#### 创建正态分布随机初始化的矩阵

当使用 `torch.randn` 函数时，你可以传递一个或多个整数作为参数，用来指定要创建的张量的形状。该函数会返回一个张量，其中的元素是从标准正态分布（均值为0，方差为1）中抽取的随机数。

以下是使用 `torch.randn` 函数的基本用法：

```python
import torch

# 创建一个形状为 (3, 3) 的张量，其中的元素是从标准正态分布中抽取的随机数
random_tensor = torch.randn(3, 3)
print(random_tensor)
```

这将会输出一个形状为 (3, 3) 的张量，其中的元素是从标准正态分布中抽取的随机数。例如：

```
tensor([[ 0.2173, -1.1514,  0.8729],
        [-0.3156,  0.9732, -0.6421],
        [ 0.0595,  0.1926, -0.7757]])
```

你也可以传递一个包含整数的元组作为参数，来指定更高维度的张量。例如，`torch.randn(2, 3, 4)` 将会创建一个形状为 (2, 3, 4) 的张量，其中的元素是从标准正态分布中抽取的随机数。



#### 创建一个全为零的矩阵并且数据元素的类型为long

你可以使用`torch.zeros`函数创建一个全为零的矩阵，并通过参数`dtype`指定数据元素的类型为`torch.long`。以下是一个示例代码：

```python
import torch

# 创建一个大小为3x3的全零矩阵，数据元素类型为long
zero_matrix = torch.zeros(3, 3, dtype=torch.long)

# 输出全零矩阵
print("全零矩阵:")
print(zero_matrix)
```

在这个例子中，`torch.zeros`函数创建了一个大小为3x3的全零矩阵，而参数`dtype=torch.long`指定了数据元素的类型为`long`。这样，矩阵的所有元素都将被初始化为零，并且其数据类型为`long`。

#### 创建一个全为一的矩阵

当然，你可以使用`torch.ones`函数创建一个全为一的矩阵，并通过参数`dtype`指定数据元素的类型。以下是一个示例代码：

```python
import torch

# 创建一个大小为4x2的全一矩阵，数据元素类型为float
ones_matrix = torch.ones(4, 2, dtype=torch.float)

# 输出全一矩阵
print("全一矩阵:")
print(ones_matrix)
```

在这个例子中，`torch.ones`函数创建了一个大小为4x2的全一矩阵，而参数`dtype=torch.float`指定了数据元素的类型为`float`。这样，矩阵的所有元素都将被初始化为一，并且其数据类型为`float`。

#### 额外解释

`torch.empty(3, 3)`和`torch.zeros(3, 3)`可能都是

```cpp
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
```

虽然在具体数值上它们看起来相似，但`torch.empty(3, 3)`和`torch.zeros(3, 3)`之间有一些重要的区别。

1. **初始化方式：**
   - `torch.empty(3, 3)`：创建一个未初始化的矩阵，矩阵中的数值是未定义的，可能包含之前在内存中存储的任意值。
   - `torch.zeros(3, 3)`：创建一个全为零的矩阵，所有元素都被明确初始化为零。

2. **性能：**
   - `torch.empty`比`torch.zeros`稍微更快，因为它不需要初始化元素为零。然而，这种差异在实践中可能不会对大多数应用产生显著的影响。

在实践中，如果你需要一个全为零的矩阵，推荐使用`torch.zeros`，因为它更加明确，且代码更易读。如果你需要一个未初始化的矩阵，并且打算在之后的代码中覆盖它的值，可以使用`torch.empty`。

#### 指定数值的矩阵
你可以使用`torch.full`函数创建一个指定数值的矩阵。以下是一个简单的例子：

```python
import torch

# 创建一个大小为2x3的矩阵，所有元素都初始化为7
specified_value_matrix = torch.full((2, 3), fill_value=7)

# 输出指定数值的矩阵
print("指定数值的矩阵:")
print(specified_value_matrix)
```

在这个例子中，`torch.full`函数创建了一个大小为2x3的矩阵，其中所有元素的值都被初始化为指定的数值（这里是7）。你可以通过调整`fill_value`参数来指定不同的数值。这种方式可以方便地创建包含相同值的矩阵。

#### 预先规定数值的矩阵

如果你想要创建一个包含不同预先规定数值的矩阵，你可以直接使用`torch.tensor`函数，并传递一个包含你想要的值的列表或嵌套列表。以下是一个例子：

```python
import torch

# 创建一个3x3的矩阵，指定不同的数值
custom_matrix = torch.tensor([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

# 输出自定义数值的矩阵
print("自定义数值的矩阵:")
print(custom_matrix)
```

在这个例子中，我们通过`torch.tensor`函数创建了一个3x3的矩阵，其中包含了我们指定的不同数值。你可以根据需要修改矩阵的大小和数值。这种方法非常直观，适用于手动指定矩阵元素值的情况。

#### 张量的概念

张量是一个非常核心的概念，尤其是在深度学习和科学计算领域。在数学中，张量是一个可以看作是向量和矩阵概念的扩展。它是一个可以在多个维度上表示数据的容器。在不同的上下文中，张量可以有不同的含义：

1. **零维张量（0D 张量）**：
   零维张量是一个单一的数字，也被称为标量（Scalar）。在 PyTorch 中，一个标量可以表示为 `torch.tensor(1)`。

2. **一维张量（1D 张量）**：
   一维张量是数字的线性阵列，通常被称为向量（Vector）。例如，`torch.tensor([1, 2, 3])` 是一个一维张量。

3. **二维张量（2D 张量）**：
   二维张量是数字的矩阵，通常用于表示传统的矩阵（Matrix）。例如，`torch.tensor([[1, 2], [3, 4]])` 是一个二维张量。

4. **三维张量（3D 张量）**：
   三维张量可以被看作是矩阵的堆叠。在深度学习中，三维张量常用于表示序列数据，如时间序列或文本数据，其中每个矩阵可以表示一个数据点的特征。例如，`torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])` 是一个三维张量。

5. **更高维度的张量**：
   张量可以扩展到任意数量的维度。在深度学习中，常见的是四维张量，特别是在处理图像数据时（例如，一个四维张量可能用于表示一批图像数据，其中维度分别对应于批次大小、通道数、高度和宽度）。

```python
x = torch.rand(2, 3, 4)
print(x)
'''tensor([[[0.9656, 0.4983, 0.8756, 0.4015],
         [0.2490, 0.3317, 0.0988, 0.7471],
         [0.3171, 0.1603, 0.0714, 0.6397]],

        [[0.5812, 0.5478, 0.0759, 0.4036],
         [0.8674, 0.4039, 0.7710, 0.7838],
         [0.8423, 0.1421, 0.1826, 0.3398]]])'''
```



在 PyTorch 中，张量是库的基础，用于表示所有类型的数据。张量可以在 CPU 或 GPU 上创建和操作，允许进行高效的科学计算。通过这种方式，张量提供了一种统一的接口来处理各种不同类型和复杂度的数据。

#### 如何理解张量的维度


理解多维张量的关键在于把它们看作是嵌套的数据结构。一个 `torch.rand(2, 3, 4)` 的张量可以被看作是一个有 2 个元素的数组，其中每个元素是一个形状为 `3x4` 的矩阵。

让我们逐步解析这个张量：

1. **最外层维度（2）**： 这个张量的最外层有 2 个元素。在这个例子中，你可以想象它为一个有两个格子的容器，每个格子中都装着一个 `3x4` 的矩阵。
2. **中间层维度（3）**： 每个 `3x4` 矩阵有 3 行。所以，在每个格子中，你有 3 行数据。
3. **最内层维度（4）**： 每行有 4 个元素。因此，在每行中，你有 4 列数据。

这样，你的张量就像是一个书架，其中有 2 个不同的隔间（最外层维度），每个隔间有 3 个不同的书架层（中间层维度），而每层书架上放着 4 本书（最内层维度）。
![image-20240206103758915](.\assets\image-20240206103758915.png)

或者可以这样理解，**4表示一个四元素的向量，例如[0.9656, 0.4983, 0.8756, 0.4015]，3表示这样的向量出现3次，构成一个矩阵，2表示这样的矩阵出现2次。如果前面还有数字，就是这样的结构又出现若干次。**

在处理高维张量时，通常需要关注每个维度的含义，这在深度学习中尤为重要。例如，在处理图像数据时（假设使用的是常见的 `批量大小 x 通道数 x 高度 x 宽度` 的格式），每个维度代表了不同的数据特征：

- **批量大小**：一次处理的图像数量。
- **通道数**：颜色通道的数量，例如，RGB 图像有 3 个通道。
- **高度和宽度**：图像的尺寸。

#### 张量尺寸

在PyTorch中，你可以使用`size()`方法来获取张量的尺寸，并使用`torch.zeros_like()`或`torch.ones_like()`等函数构建具有相同尺寸的新张量。以下是一个示例：

```python
import torch

# 创建一个大小为3x4的张量
original_tensor = torch.rand(3, 4)

# 获取张量的尺寸
tensor_size = original_tensor.size()

# 构建一个全零的张量，尺寸与原始张量相同
zero_tensor = torch.zeros_like(original_tensor)

# 构建一个全一的张量，尺寸与原始张量相同
ones_tensor = torch.ones_like(original_tensor)

# 输出结果
print("原始张量:")
print(original_tensor)
print("原始张量的尺寸:", tensor_size)
print("全零张量:")
print(zero_tensor)
print("全一张量:")
print(ones_tensor)
```

在这个例子中，`original_tensor`是一个大小为3x4的张量，使用`size()`方法获取了它的尺寸。然后，通过`torch.zeros_like()`和`torch.ones_like()`构建了两个与`original_tensor`相同尺寸的全零和全一张量。
#### 一行代码内获取一个矩阵的行数和列数


你可以使用`shape`属性一行代码内获取一个矩阵的行数和列数，并将它们分别赋给两个变量。以下是一个例子：

```python
import torch

# 创建一个3x4的矩阵
matrix = torch.rand(3, 4)

# 一行代码获取行数和列数并赋值给两个变量
rows, columns = matrix.shape

# 输出结果
print("矩阵:")
print(matrix)
print("行数:", rows)
print("列数:", columns)
```

在这个例子中，`matrix.shape`返回一个包含行数和列数的元组，通过解构赋值一行代码内将这两个值分别赋给了`rows`和`columns`两个变量。
#### 四种常见的矩阵加法


矩阵加法是一种按元素进行相加的运算，对应位置上的元素相加。在PyTorch中，有四种常见的矩阵加法，包括矩阵与矩阵的加法、矩阵与标量的加法、逐元素加法和原地加法。

1. **矩阵与矩阵的加法：**
   ```python
   import torch
   
   # 创建两个矩阵
   matrix1 = torch.rand(2, 3)
   matrix2 = torch.rand(2, 3)
   
   # 矩阵与矩阵的加法
   result_matrix = matrix1 + matrix2
   
   print("matrix1: ", matrix1)
   print("matrix2: ", matrix2)
   print("result_matrix: ", result_matrix)
   ```
   
2. **矩阵与标量的加法：**
   ```python
   import torch
   
   # 创建一个矩阵
   matrix = torch.rand(3, 3)
   
   # 矩阵与标量的加法
   scalar = 2.0
   result_matrix = matrix + scalar
   print("result_matrix: ", result_matrix)
   ```
   
3. **逐元素加法：**
   ```python
   import torch
   
   # 创建两个矩阵
   matrix1 = torch.rand(2, 3)
   matrix2 = torch.rand(2, 3)
   
   # 逐元素加法
   result_matrix = torch.add(matrix1, matrix2)
   
   print("matrix1: ", matrix1)
   print("matrix2: ", matrix2)
   print("result_matrix: ", result_matrix)
   ```
   
4. **原地加法：**
   ```python
   import torch
   
   # 创建两个矩阵
   matrix1 = torch.rand(2, 3)
   matrix2 = torch.rand(2, 3)
   
   print("matrix1: ", matrix1)
   print("matrix2: ", matrix2)
   
   # 原地加法，将结果存储在matrix1中
   matrix1.add_(matrix2)
   
   print("matrix1: ", matrix1)
   ```

这四种加法方式分别适用于不同的情境，选择合适的加法方式取决于你的具体需求。原地加法会直接修改原始矩阵，而其他方式会创建一个新的矩阵来存储结果。
#### 获取矩阵的某一列、某一行或者一个子矩阵

在PyTorch中，你可以使用类似于NumPy的索引和切片操作来获取矩阵的某一列、某一行或者一个子矩阵。这里是一些基本的示例：

1. **获取一行**：
   要获取矩阵的某一行，你可以使用索引。例如，要获取第 `i` 行，你可以使用 `matrix[i]`。

2. **获取一列**：
   要获取矩阵的某一列，使用冒号 `:` 来表示所有行，然后指定列的索引。例如，要获取第 `j` 列，使用 `matrix[:, j]`。

3. **获取一个子矩阵**：
   你可以通过指定行和列的范围来获取子矩阵。例如，要获取从第 `i` 行到第 `k` 行，第 `j` 列到第 `l` 列的子矩阵，使用 `matrix[i:k, j:l]`。

这里是一个具体的例子：

假设我们有一个 4x4 的矩阵，我们想要获取第 2 行、第 3 列和一个位于第 2-3 行、第 1-2 列的子矩阵。

```python
import torch

# 创建一个 4x4 的矩阵
matrix = torch.tensor([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]])

# 获取第 2 行
row = matrix[1]

# 获取第 3 列
col = matrix[:, 2]

# 获取子矩阵（第 2-3 行，第 1-2 列）
sub_matrix = matrix[1:3, 0:2]

print("Row:\n", row)
print("Column:\n", col)
print("Sub-matrix:\n", sub_matrix)
```

在这个例子中，`matrix[1]` 获取第 2 行（因为索引是从 0 开始的），`matrix[:, 2]` 获取第 3 列，`matrix[1:3, 0:2]` 获取一个子矩阵，它包含第 2-3 行和第 1-2 列的元素。

#### 改变张量的形状
在PyTorch中，改变张量的形状是一个非常常见的操作，可以使用多种方法来实现。下面介绍几种常用的方法：

1. **`view()` 方法**：
   `view()` 方法用于重新塑形张量。它返回一个新的张量，其数据与原张量共享，但形状可能不同。当你使用 `view()` 时，新的形状必须与原始张量中的元素数目一致。例如，你可以将一个形状为 `[4, 5]` 的矩阵重新塑形为一个形状为 `[20]` 的向量。

   ```python
   import torch
   
   # 创建一个 4x5 的矩阵
   matrix = torch.arange(20).reshape(4, 5)
   
   # 使用 view 将其变为一个向量
   vector = matrix.view(20)
   
   # 使用 view 改变矩阵的形状为 2x10
   reshaped_matrix = matrix.view(2, 10)
   
   print(torch.arange(20))
   print(matrix)
   print(vector)
   print(reshaped_matrix)
   ```
   
2. **`reshape()` 方法**：
   `reshape()` 方法的功能与 `view()` 类似，但当原始数据不连续时，`reshape()` 可以返回一个实际的拷贝。这使得 `reshape()` 更加灵活，但有时可能效率较低。

   ```python
   # 使用 reshape 改变矩阵的形状
   reshaped_matrix = matrix.reshape(-1, 10)
   ```

3. **`flatten()` 方法**：
   `flatten()` 方法用于将张量扁平化为一维。这对于从多维结构转换到一维向量特别有用，例如在将特征图送入全连接层之前。

   ```python
   # 将矩阵扁平化为一个向量
   flattened_matrix = matrix.flatten()
   ```

4. **`squeeze()` 和 `unsqueeze()` 方法**：
   - `squeeze()` 方法用于去除张量中所有维度为 1 的维度。
   - `unsqueeze()` 方法用于在指定位置添加一个维度为 1 的维度。

   这些方法在处理包含单一维度的张量时非常有用。

   ```python
   # 假设我们有一个形状为 [1, 20] 的张量
   tensor = torch.zeros(1, 20)
   
   # 使用 squeeze 去除多余的维度
   squeezed_tensor = tensor.squeeze()
   
   # 使用 unsqueeze 添加一个新的维度
   unsqueezed_tensor = tensor.unsqueeze(0)
   ```

当我们谈论使用 `squeeze()` 方法去除张量中所有维度为 1 的维度时，我们通常是指去除那些多余的、不影响张量中元素排列的维度。
例如，假设我们有一个 2D 矩阵，其形状为 `[1, n]` 或 `[n, 1]`，其中 `n` 是元素的数量。在这种情况下，虽然矩阵在技术上是二维的，但其实际上只在一个维度上有扩展。`squeeze()` 方法可以用来去除那个单一的维度，将其转换成一个一维的向量（形状为 `[n]`）。

请注意，所有这些方法都不会改变原始张量的数据，它们只是改变了数据的视图或表示。在使用这些方法时，你需要确保所请求的新形状与原始数据的元素数目是兼容的。

**额外**

如果一个张量中只包含一个元素，你可以使用 `.item()` 方法将它作为一个标准的Python数值取出。这在提取单个值的时候非常有用，特别是在你需要将这个值用于不接受张量的Python操作时。

例如：

```python
import torch

# 创建一个只包含一个元素的张量
tensor = torch.tensor([7])

# 使用 item() 方法将其转换为一个 Python 数字
number = tensor.item()

print(number)  # 输出: 7
```

关于 `reshape()` 方法中的 `-1`，这是一个特殊的参数，表示让 PyTorch 自动计算这个维度的大小。使用 `-1` 可以让你在重新塑形张量时不必显式地指定每个维度的大小，这在你只关心某些维度的大小时特别有用。

当你在 `reshape()` 方法中使用 `-1` 时，PyTorch 会自动计算这个维度的大小，以保证总的元素数量与原始张量相同。

例如：

```python
import torch

# 创建一个 2x3 的矩阵
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 使用 reshape 并在其中一个维度上使用 -1
# 这里我们只关心将其转换为一个行数不确定的二维张量
reshaped_matrix = matrix.reshape(-1, 3)

print(reshaped_matrix)
```

在这个例子中，`matrix.reshape(-1, 3)` 会将 `matrix` 变形为一个列数为 3 的二维张量，而行数由 PyTorch 自动计算以确保元素总数不变。由于原始张量有 6 个元素，新的形状将是 `[2, 3]`。使用 `-1` 使得我们不需要手动计算行数。

在 PyTorch 中，可以使用简单的方法来在 PyTorch 张量（tensors）和 NumPy 数组（arrays）之间进行转换。这些操作通常很直观，但需要注意它们在内存共享方面的行为。
#### 基本算术运算

在 PyTorch 中，你可以进行各种类型的张量（Tensor）运算，这些运算覆盖了从基本的算术运算到复杂的线性代数运算。下面是一些常见的运算示例：

1. **矩阵加法**：
   矩阵之间的加法遵循元素对元素的规则。

   ```python
   a = torch.tensor([[1, 2], [3, 4]])
   b = torch.tensor([[5, 6], [7, 8]])
   c = a + b  # 元素对元素相加
   print(c)
   ```

2. **矩阵和标量相加**：
   矩阵的每个元素都会与标量相加。

   ```python
   a = torch.tensor([[1, 2], [3, 4]])
   c = a + 5  # 每个元素加 5
   print(c)
   ```

3. **矩阵乘法**：
   矩阵乘法可以使用 `torch.matmul` 或 `@` 运算符。

   ```python
   a = torch.tensor([[1, 2], [3, 4]])
   b = torch.tensor([[5, 6], [7, 8]])
   c = torch.matmul(a, b)  # 或者 c = a @ b
   print(c)
   ```

![image-20240206000643472](.\assets\image-20240206000643472.png)

4. **矩阵乘向量**

   ```py
   import torch
   
   matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
   vector = torch.tensor([7, 8, 9])
   
   print(matrix@vector)#tensor([ 50, 122])
   print(vector@matrix.T)#tensor([ 50, 122])
   ```

   <img src="./assets/image-20240317165239536.png" alt="image-20240317165239536" style="zoom:67%;" />

   为什么两种写法都可以

   1. **`matrix @ vector`**：这种写法中，我们有一个矩阵`matrix`和一个向量`vector`进行乘法操作。按照矩阵乘法的规则，`matrix`的列数必须与`vector`的行数相匹配。在这个例子中，`matrix`是一个2×3矩阵，`vector`可以被视为一个3×1的矩阵（虽然在PyTorch中它是一维的，但在矩阵乘法中可以这样理解）。因此，这个操作是合法的，并且结果是一个2×1的向量（在PyTorch中表示为一维向量，长度为2）。
   2. **`vector @ matrix.T`**：在这种写法中，`vector`首先与`matrix.T`（`matrix`的转置）进行乘法操作。`matrix.T`是一个3×2的矩阵，与`vector`的维度（视为1×3矩阵）相匹配。这种情况下，结果也是一个向量，其形状为1×2（在PyTorch中表示为一维向量，长度为2）。
   3. **一个矩阵乘以一个向量，在pytorch的表示上，得到的是一个向量，而不是一个矩阵。**[后面会意识到这一点]

1. **逐元素乘法（哈达玛积）**：
   两个矩阵的逐元素乘法。

   ```python
   a = torch.tensor([[1, 2], [3, 4]])
   b = torch.tensor([[5, 6], [7, 8]])
   c = a * b  # 逐元素相乘
   print(c)
   ```

2. **矩阵逐元素除法**：
   矩阵的逐元素除法。

   ```python
   a = torch.tensor([[1, 2], [3, 4]])
   b = torch.tensor([[5, 6], [7, 8]])
   c = a / b  # 逐元素相除
   print(c)
   ```

3. **矩阵和标量相除**：
   矩阵的每个元素都会被标量除。

   ```python
   a = torch.tensor([[1, 2], [3, 4]])
   c = a / 5  # 每个元素除以 5
   print(c)
   c =  5 /a  
   print(c)
   ```

4. **矩阵减法**：
   矩阵之间的减法遵循元素对元素的规则。

   ```python
   a = torch.tensor([[1, 2], [3, 4]])
   b = torch.tensor([[5, 6], [7, 8]])
   c = a - b  # 元素对元素相减
   print(c)
   ```

#### PyTorch的广播机制

PyTorch的广播机制允许在不同维度之间进行运算，使得在形状不完全匹配的张量上执行逐元素操作成为可能。广播操作包括两个步骤：扩展（Broadcasting）和逐元素操作（Element-wise operation）。下面通过一个具体的例子来详细说明PyTorch中的广播机制：

```python
import torch

# 创建两个张量
tensor1 = torch.tensor([[1, 2, 3],
                        [4, 5, 6]])  # 形状为 (2, 3)
tensor2 = torch.tensor([10, 20, 30])    # 形状为 (3)

# 执行加法操作
result = tensor1 + tensor2

print("Tensor 1:")
print(tensor1)
print("\nTensor 2:")
print(tensor2)
print("\nResult after broadcasting:")
print(result)
```

在这个例子中，我们有一个形状为(2, 3)的二维张量`tensor1`和一个形状为(3,)的一维张量`tensor2`。在加法操作中，`tensor2`的维度被扩展以匹配`tensor1`的形状，然后对应位置的元素进行加法操作。这就是广播机制的工作原理。

具体来说，PyTorch首先将`tensor2`的形状扩展为(2, 3)，使其与`tensor1`的形状匹配。然后，对应位置的元素进行加法操作：

```
tensor1:
[[1, 2, 3],
 [4, 5, 6]]

tensor2 (broadcasted to [[10, 20, 30],
                         [10, 20, 30]]):

[[10, 20, 30],
 [10, 20, 30]]

Result after broadcasting:

[[11, 22, 33],
 [14, 25, 36]]
```

通过广播机制，我们可以在形状不完全匹配的张量上执行逐元素操作，使得代码更加简洁和高效。
**之所以张量可以和标量进行运算，本质上是一种特殊的广播机制，即：将标量复制成张量。**

#### 高级运算

1. **乘方**：
   计算矩阵的每个元素的乘方。

   ```python
   a = torch.tensor([[1, 2], [3, 4]])
   c = a ** 2  # 每个元素的平方
   print(c)
   c = a ** 3.1  
   print(c)
   
   ```

2. **矩阵与向量的乘法**：
   当用一个矩阵乘以一个向量时，进行的是矩阵乘法。

   ```python
   a = torch.tensor([[1, 2], [3, 4]])
   v = torch.tensor([1, 2])
   c = torch.matmul(a, v)  # 矩阵和向量的乘法
   print(c)
   c= a @ v
   print(c)
   ```

   ![image-20240206100829147](.\assets\image-20240206100829147.png)

3. **向量内积（点积）**：
   两个向量的内积或点积。

   ```python
   v1 = torch.tensor([1, 2, 3])
   v2 = torch.tensor([4, 5, 6])
   dot_product = torch.dot(v1, v2)
   print(dot_product)#4+10+18=32
   ```

4. **转置**：
   获取矩阵的转置。

   ```python
   a = torch.tensor([[1, 2], [3, 4]])
   a_transpose = a.t()
   ```

![image-20240206101057868](.\assets\image-20240206101057868.png)

1. **逆矩阵**：


 计算可逆矩阵的逆。

   ```python
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
a_inverse = torch.inverse(a)
print(a_inverse)
#tensor([[-2.0000,  1.0000],
#        [ 1.5000, -0.5000]])
   ```

![image-20240206101901034](.\assets\image-20240206101901034.png)
这些只是 PyTorch 提供的众多张量运算中的一小部分。PyTorch 张量支持大多数在 NumPy 中找到的运算，并且很多运算可以在 GPU 上执行，以提高计算效率。在实际的深度学习应用中，这些运算是构建和训练神经网络的基础。|

PyTorch 提供的张量运算不仅限于基本的算术运算，还包括更多高级和专门的运算，这些运算在不同的应用场景（如深度学习、科学计算等）中非常有用。以下是一些其他常见的张量运算的例子：

#### 统计运算

1. **求和**：
   计算张量的所有元素的总和。
   ```python
   a = torch.tensor([[1, 2], [3, 4]])
   sum = torch.sum(a)
   print(sum)
   ```
   
2. **平均值（均值）**：
   计算张量的所有元素的平均值。
   ```python
   mean = torch.mean(a.float())  # 注意 mean 需要浮点数输入
   print(mean)
   ```
   
3. **最大值和最小值**：
   寻找张量中的最大值或最小值。
   ```python
   max_val = torch.max(a)
   min_val = torch.min(a)
   print(max_val,min_val)
   ```
   
4. **标准差和方差**：
   计算张量的标准差和方差。
   ```python
   import math
   std_dev = torch.std(a.float())  # 需要浮点数输入
   #std_dev 是 "Standard Deviation" 的缩写
   variance = torch.var(a.float())
   print(std_dev,variance)
   print(math.sqrt(variance))
   ```

在 PyTorch 中，`torch.std()` 和 `torch.var()` 默认计算的是样本标准差和样本方差，而不是总体标准差和总体方差。样本标准差和样本方差在计算时使用的是 Bessel's correction（贝塞尔校正），即分母使用 `n-1` 而不是 `n`.
![image-20240206102905942](.\assets\image-20240206102905942.png)

在 PyTorch 中，`torch.var()` 函数用于计算张量的方差。该函数有一个参数 `unbiased`，其作用是决定计算方差时是否使用无偏估计（Bessel's correction）。`unbiased` 参数的默认值是 `True`。

当 `unbiased=True` 时，`torch.var()` 使用无偏估计来计算方差，公式如下：
$$ \text{Var}(x) = \frac{1}{N - 1} \sum_{i=1}^{N} (x_i - \bar{x})^2 $$
这里，$ N $ 是样本数量，$ x_i $ 是单个样本值，$ \bar{x} $ 是样本均值。注意**分母是 $ N - 1 $**，这是 Bessel's correction，用于在有限样本情况下更准确地估计总体方差。

当 `unbiased=False` 时，`torch.var()` 使用有偏估计来计算方差，公式如下：
$$ \text{Var}(x) = \frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2 $$
在这种情况下，**分母是 $ N $**，这意味着它是基于样本本身的方差而非总体方差的估计。

选择使用无偏估计还是有偏估计取决于你的具体应用和需要。在统计学中，当样本数量较少时，无偏估计通常被认为是更准确的，因为它修正了由于样本数量有限而导致的估计偏差。然而，在深度学习和其他计算密集型任务中，有时会选择有偏估计，因为它在计算上更简单且对于大数据集来说差异通常很小。

#### 在指定维度进行统计(求softmax的时候很有用)

当我们在PyTorch中使用`torch.sum()`函数时，通过指定`dim`参数可以沿着指定的维度对张量进行求和。下面是一个具体的示例，说明如何沿着某个维度对张量进行求和：

```python
import torch

# 创建一个3维张量
tensor = torch.tensor([
    [[1, 2, 3],
     [4, 5, 6]],
    
    [[7, 8, 9],
     [10, 11, 12]]
])

# 沿着第一个维度（索引为0的维度）求和
sum_along_dim0 = torch.sum(tensor, dim=0)
print("Sum along dim 0:")
print(sum_along_dim0)
print()
'''tensor([[ 8, 10, 12],
        [14, 16, 18]])'''

# 沿着第二个维度（索引为1的维度）求和
sum_along_dim1 = torch.sum(tensor, dim=1)
print("Sum along dim 1:")
print(sum_along_dim1)
print()
'''tensor([[ 5,  7,  9],
        [17, 19, 21]])'''

# 沿着第三个维度（索引为2的维度）求和
sum_along_dim2 = torch.sum(tensor, dim=2)
print("Sum along dim 2:")
print(sum_along_dim2)
'''tensor([[ 6, 15],
        [24, 33]])'''
```

![image-20240209150036185](.\assets\image-20240209150036185.png)
这段代码创建了一个形状为(2, 2, 3)的3维张量，并对其进行了沿着不同维度的求和操作。`torch.sum()`函数的`dim`参数指定了要进行求和的维度，保持维度的方式通过`keepdim=True`来设置。在这个例子中，我们对不同的维度进行了求和，分别沿着第一个、第二个和第三个维度进行了求和。

**额外参数keepdim=True**
`keepdim=True`时，结果张量将保持与原始张量相同的维度数，只是在指定的维度上长度为1。这样做的好处是可以保持结果张量与原始张量的形状一致，方便后续的广播或其他操作。

```python
# 沿着第二个维度（索引为1的维度）求和
sum_along_dim1 = torch.sum(tensor, dim=1,keepdim=True)
print("Sum along dim 1:")
print(sum_along_dim1)
print()

Sum along dim 1:
tensor([[[ 5,  7,  9]],
        [[17, 19, 21]]])
```

#### 更复杂的指定维度进行统计的例子:

```python
import torch
import torch.nn as nn
input_tensor = torch.tensor([[[[1, 2, 3, 4],
                               [5, 6, 7, 8],
                               [9, 10, 11, 12],
                               [13, 14, 15, 16]],

                              [[17, 18, 19, 20],
                               [21, 22, 23, 24],
                               [25, 26, 27, 28],
                               [29, 30, 31, 32]],

                              [[33, 34, 35, 36],
                               [37, 38, 39, 40],
                               [41, 42, 43, 44],
                               [45, 46, 47, 48]]],


                             [[[49, 50, 51, 52],
                               [53, 54, 55, 56],
                               [57, 58, 59, 60],
                               [61, 62, 63, 64]],

                              [[65, 66, 67, 68],
                               [69, 70, 71, 72],
                               [73, 74, 75, 76],
                               [77, 78, 79, 80]],

                              [[81, 82, 83, 84],
                               [85, 86, 87, 88],
                               [89, 90, 91, 92],
                               [93, 94, 95, 96]]]]).float()

mean = input_tensor.mean([0, 2, 3])
print(mean)
'''tensor([32.5000, 48.5000, 64.5000])'''

```

![image-20240213192448294](.\assets\image-20240213192448294.png)

在第3个维度压缩, 我们就应该把所有的在第3个维度坐标相同的数值加在一起,而第3个维度坐标相同的数的组合例如[1, 2, 3, 4]，[81, 82, 83, 84]。
 在第2个维度压缩。 我们就应该把所有的在第2个维度坐标相同的数值加在一起，而第2个维度坐标相同的数的组合例如[1, 5, 9, 13],[49,53,57,61] 得到的结果形如:[[,,],[,,]]
最后把所有的在第0个维度坐标相同的数值加在一起，最终结果就形如[,,]

大概理解一下即可。

#### 形状和布局变换

1. **重塑（Reshape）**：
   改变张量的形状。
   ```python
   reshaped = a.reshape(1, 4)
   print(reshaped)
   ```
   
2. **展开（Flatten）**：
   将张量展平为一维。
   
   ```python
   flattened = a.flatten()
   print(flattened)
   ```
   
3. **交换维度（Permute）**：
   改变张量的维度顺序。
   ```python
   x = torch.rand(2, 3, 4)
   permuted = x.permute(2, 0, 1)  # 改变维度的顺序
   print(x.permute(0,1,2))
   print(permuted)
   ```

在 PyTorch 中，`permute` 方法用于重新排列张量的维度。当你对一个张量使用 `permute` 方法时，你需要指定新的维度顺序。在你的例子中，`x` 是一个形状为 `(2, 3, 4)` 的张量，而 `permute(2, 0, 1)` 会将这个张量的维度重新排列。

具体来说：

- 原始张量 `x` 的维度是 `(2, 3, 4)`，即它有 2 个 `3x4` 的矩阵。
- 当你调用 `permuted = x.permute(2, 0, 1)` 时，你将 `x` 的维度重新排列为 `(4, 2, 3)`。

这意味着：

- 第一个维度（`2`）被移到了中间位置。
- 第二个维度（`3`）被移到了最后位置。
- 第三个维度（`4`）被移到了第一个位置。

所以，如果你想象原始的 `x` 是一个装有 2 个 `3x4` 矩阵的盒子，`permute` 操作后得到的 `permuted` 张量就像是一个装有 4 个 `2x3` 矩阵的盒子。
![image-20240206105143205](.\assets\image-20240206105143205.png)

这种维度的重排列在处理具有特定维度要求的操作时非常有用，比如在深度学习中处理具有不同数据格式（比如从 NHWC 到 NCHW）的图像数据时。

#### 高级线性代数运算

1. **矩阵的特征值和特征向量**：
   计算方阵的特征值和特征向量。

   ```python
   # 创建一个张量
   a = torch.tensor([[0.0, -1.0], [1.0,0.0]])
   
   # 使用 torch.linalg.eig 计算特征值和特征向量
   eigvals, eigvecs = torch.linalg.eig(a)
   
   print("Eigenvalues:", eigvals)
   print("Eigenvectors:", eigvecs)
   '''Eigenvalues: tensor([0.+1.j, 0.-1.j])
   Eigenvectors: tensor([[0.7071+0.0000j, 0.7071-0.0000j],
           [0.0000-0.7071j, 0.0000+0.7071j]])'''
   ```

考虑矩阵：

$$
A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
$$

对于矩阵 $A$，我们求特征方程 $$det(A - \lambda I) = 0$$ 的解：

$$
\begin{vmatrix} -\lambda & -1 \\ 1 & -\lambda \end{vmatrix} = \lambda^2 + 1 = 0
$$

解这个方程，我们得到 $\lambda^2 = -1$，因此 $\lambda = \pm i$，其中 $i$ 是虚数单位。因此，这个矩阵的特征值是纯虚数。

#### 特征向量的求解

首先，我们来看看如何手动求解特征向量。对于矩阵 $ A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} $ 和特征值 $ \lambda = i $ 和 $ \lambda = -i $，特征向量可以通过解线性方程组 $ (A - \lambda I) \mathbf{v} = 0 $ 来找到，其中 $ \mathbf{v} $ 是特征向量。

例如，对于 $ \lambda = i $：

1. 构造矩阵 $ A - iI $：
   $$
   A - iI = \begin{bmatrix} 0 - i & -1 \\ 1 & 0 - i \end{bmatrix} = \begin{bmatrix} -i & -1 \\ 1 & -i \end{bmatrix}
   $$
   
2. 解方程 $ (A - iI) \mathbf{v} = 0 $ 以找到 $ \mathbf{v} $。

类似的步骤也适用于 $ \lambda = -i $。
![image-20240206114444888](.\assets\image-20240206114444888.png)

#### PyTorch 计算的特征向量

在 PyTorch 中，求得的特征向量是标准化的，这意味着它们的长度（或范数）被归一化为 1。在你的例子中，特征向量是复数，且其模长（绝对值）被标准化为 1。

以第一个特征向量为例：`[0.7071+0.0000j, 0.0000-0.7071j]`。这个向量的模长为：

$$
 \sqrt{(0.7071)^2 + (-0.7071)^2} = \sqrt{0.5 + 0.5} = \sqrt{1} = 1 
$$


所以，PyTorch 返回的特征向量是单位特征向量。它选择 $ 0.7071 $（即 $ \frac{1}{\sqrt{2}} $）是因为这样可以使特征向量的模长为 1，满足标准化的要求。这是数值计算中常用的做法，以保证结果的一致性和可比较性。
![image-20240206114106999](.\assets\image-20240206114106999.png)

请注意，特征向量通常不是唯一的，因为任何非零标量倍的特征向量也是一个有效的特征向量。在实际应用中，通常会选择某种标准化形式，比如使其模长为 1。


1. **奇异值分解（SVD）**：
   对矩阵进行奇异值分解。

   ```python
   # 创建一个张量
   a = torch.tensor([[0.0, -1.0], [1.0,0.0]])
   u, s, v = torch.svd(a.float())
   '''tensor([[ 0., -1.],
           [-1.,  0.]])
   tensor([1., 1.])
   tensor([[-1.,  0.],
           [-0.,  1.]])'''
   ```

为了验证奇异值分解（SVD）的正确性，需要证明原始矩阵 $ A $ 可以通过分解得到的 $ U $，$ \Sigma $（奇异值矩阵），和 $ V^T $ 重构出来。换句话说，你需要验证以下等式是否成立：

$$ A = U \Sigma V^T $$

让我们用你提供的例子来进行验证：

给定的矩阵 $ A $ 和 SVD 分解结果为：

- $ A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} $
- $ U = \begin{bmatrix} 0 & -1 \\ -1 & 0 \end{bmatrix} $
- $ \Sigma = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} $
- $ V^T = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix} $

首先，构造奇异值矩阵 $ \Sigma $ 的完整形式（与 $ A $ 有相同的形状）：

$$ \Sigma_{full} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} $$

然后，计算 $ U \Sigma V^T $：

$$ U \Sigma_{full} V^T = \begin{bmatrix} 0 & -1 \\ -1 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix} $$

执行矩阵乘法：

$$ = \begin{bmatrix} 0 & -1 \\ -1 & 0 \end{bmatrix} \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix} $$
$$ = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} $$

可以看到，这正好是原始矩阵 $ A $。因此，SVD 分解是正确的。

**注:**
在奇异值分解（SVD）中，通常会得到一个包含奇异值的一维数组或向量。为了在重构原始矩阵时使用这些奇异值，你需要首先将这个数组转换成一个对角矩阵，这个对角矩阵通常被称为 $ \Sigma $。然后，为了使其与原始矩阵的维度相匹配，你可能需要将 $ \Sigma $ 扩展为一个更大的矩阵，称为 $ \Sigma_{\text{full}} $。
![image-20240206120334406](.\assets\image-20240206120334406.png)

**如何构造 $ \Sigma $**

假设你有一个奇异值向量 $ s = [s_1, s_2, \ldots, s_n] $，你可以将其转换为对角矩阵 $ \Sigma $，其中对角线上的元素是奇异值，其余元素都是 0。

例如，对于奇异值向量 $ s = [1, 1] $，对应的 $ \Sigma $ 为：

$$ \Sigma = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} $$

**如何构造 $ \Sigma_{\text{full}} $**

如果原始矩阵 $ A $ 是一个 $ m \times n $ 矩阵，且 $ m \neq n $，那么你需要将 $ \Sigma $ 扩展为一个 $ m \times n $ 矩阵，称为 $ \Sigma_{\text{full}} $。在 $ \Sigma_{\text{full}} $ 中，除了 $ \Sigma $ 的对角线元素外，其余元素都是 0。

例如，如果 $ A $ 是一个 $ 2 \times 3 $ 矩阵，且 $ \Sigma $ 是一个 $ 2 \times 2 $ 矩阵，那么 $ \Sigma_{\text{full}} $ 将是一个 $ 2 \times 3 $ 矩阵：

$$ \Sigma_{\text{full}} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix} $$

在你的示例中，原始矩阵 $ A $ 是一个 $ 2 \times 2 $ 矩阵，所以 $ \Sigma $ 已经是 $ \Sigma_{\text{full}} $。如果原始矩阵的尺寸不同，你需要相应地调整 $ \Sigma $ 的大小。

#### 索引、切片和连接

1. **索引和切片**：
   提取张量的特定部分。
   ```python
   a = torch.tensor([[1, 2], [3,4]])
   sliced = a[:, 1]  # 提取第二列
   print(sliced)
   ```
   
2. **连接和堆叠**：
   将多个张量在特定维度上合并。
   
   ```python
   a = torch.tensor([[1, 2], [3,4]])
   b = torch.tensor([[5, 6], [7, 8]])
   concatenated = torch.cat([a, b], dim=0)  # 纵向连接
   stacked = torch.stack([a, b], dim=0)  # 堆叠
   print(concatenated)
   print(stacked)
   '''tensor([[1, 2],
           [3, 4],
           [5, 6],
           [7, 8]])
   tensor([[[1, 2],
            [3, 4]],
   
           [[5, 6],
            [7, 8]]])'''
   ```

![image-20240206121133197](.\assets\image-20240206121133197.png)

#### 激活函数和其他非线性函数

![sigmoid_functions](.\assets\sigmoid_functions.jpg)

在深度学习中，激活函数是神经网络的一个关键组成部分，它们通常被用来引入非线性，使得网络能够学习和执行更复杂的任务。PyTorch 提供了许多常用的激活函数。以下是一些常见的激活函数及其在 PyTorch 中的使用方法：

#### 1. ReLU (Rectified Linear Unit)

对矩阵使用激活函数ReLU，相当于逐元素使用激活函数

ReLU 是最常用的激活函数之一，定义为$  \text{ReLU}(x) = \max(0, x) $。

```python
a = torch.tensor([[1, -2], [3,-4]])
relu = torch.relu(a.float())
print(relu)
'''tensor([[1., 0.],
        [3., 0.]])'''
```

![image-20240206163050782](.\assets\image-20240206163050782.png)

#### 2. Sigmoid

对矩阵使用激活函数Sigmoid，相当于逐元素使用激活函数

Sigmoid 函数将输入压缩到 0 和 1 之间，定义为 $\sigma(x) = \frac{1}{1 + e^{-x}} $。

```python
a = torch.tensor([[-100, 0], [1,100]])
sigmoid = torch.sigmoid(a.float())
print(sigmoid)
'''tensor([[0.0000, 0.5000],
        [0.7311, 1.0000]])'''
```

![image-20240206163116364](.\assets\image-20240206163116364.png)

#### 3. Tanh (Hyperbolic Tangent)

对矩阵使用激活函数Tanh ，相当于逐元素使用激活函数

Tanh 函数输出范围在 -1 到 1 之间，定义为 $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $。

```python
a = torch.tensor([[-100, 0], [1,100]])
tanh = torch.tanh(a.float())
print(tanh)
'''tensor([[-1.0000,  0.0000],
        [ 0.7616,  1.0000]])'''
```

![image-20240206163212678](.\assets\image-20240206163212678.png)

#### 4. Softmax

Softmax 函数通常用于多类分类问题的输出层，将输入转换为概率分布。对于向量 $ x $，每个元素的 Softmax 值定义为 $ \text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} $。

```python
a = torch.tensor([[-100, 0], [1,100]])
softmax = torch.softmax(a.float(), dim=1)
print(softmax)
'''tensor([[3.7835e-44, 1.0000e+00],
        [1.0089e-43, 1.0000e+00]])'''
```

当你在 `dim=1` （按行）应用 softmax 时，每行的元素被转换为概率，使得每行的概率总和为 1。

![image-20240206164324072](.\assets\image-20240206164324072.png)

```python
a = torch.tensor([[-100, 0], [1,100]])
softmax = torch.softmax(a.float(), dim=0)
print(softmax)
'''tensor([[1.4013e-44, 3.7835e-44],
        [1.0000e+00, 1.0000e+00]])'''
```

当你在 `dim=0` （按列）应用 softmax 时，每列的元素被转换为概率，使得每列的概率总和为 1。
![image-20240206164521479](.\assets\image-20240206164521479.png)

#### 5. Leaky ReLU

Leaky ReLU 是 ReLU 的一个变体，允许负输入有一个小的正斜率，定义为 $\text{LeakyReLU}(x) = \max(0.01x, x) $。

```python
a = torch.tensor([[1, -2], [3,-4]])
leaky_relu = torch.nn.functional.leaky_relu(a.float(), negative_slope=0.01)
print(leaky_relu)
'''tensor([[ 1.0000, -0.0200],
        [ 3.0000, -0.0400]])'''
```

![image-20240206165432886](.\assets\image-20240206165432886.png)

#### 6. ELU (Exponential Linear Unit)

ELU 是另一个 ReLU 的变体，对于负输入有一个指数衰减，定义为：

$
\text{ELU}(x) = 
\begin{cases} 
x & \text{if } x \geq 0 \\
\alpha(e^x - 1) & \text{if } x < 0 
\end{cases}
$

```python
a = torch.tensor([[1, -2], [3,-4]])
elu = torch.nn.functional.elu(a.float(), alpha=1.0)
print(elu)
'''tensor([[ 1.0000, -0.8647],
        [ 3.0000, -0.9817]])'''
```

![image-20240206170111803](.\assets\image-20240206170111803.png)
这些激活函数在 PyTorch 中都可以直接使用，而且它们是神经网络设计中不可或缺的工具。通过选择合适的激活函数，可以影响网络的学习和性能。

#### 从 PyTorch Tensor 转换为 NumPy Array

要将 PyTorch 张量转换为 NumPy 数组，可以使用 `.numpy()` 方法。这个操作会创建原始张量数据的一个视图，而不是拷贝数据。

```python
import torch

# 创建一个 PyTorch 张量
tensor = torch.tensor([1, 2, 3, 4])

# 转换为 NumPy 数组
numpy_array = tensor.numpy()

print(numpy_array)
```

#### 从 NumPy Array 转换为 PyTorch Tensor

要将 NumPy 数组转换为 PyTorch 张量，可以使用 `torch.from_numpy()` 函数。这个操作同样创建了数组数据的视图，而不是拷贝数据。

```python
import numpy as np
import torch

# 创建一个 NumPy 数组
numpy_array = np.array([1, 2, 3, 4])

# 转换为 PyTorch 张量
tensor = torch.from_numpy(numpy_array)

print(tensor)
```

#### 内存共享

值得注意的是，在上述两种转换中，生成的 NumPy 数组和 PyTorch 张量共享相同的内存。这意味着，如果你改变其中一个的内容，另一个的内容也会随之改变。这种行为在处理大型数据集时非常有用，因为它避免了不必要的数据复制，从而提高了效率。

```python
# 改变 NumPy 数组
numpy_array[0] = 100

# PyTorch 张量也随之改变
print(tensor)  # 输出: tensor([100,   2,   3,   4])

# 改变 PyTorch 张量
tensor[1] = 200

# NumPy 数组也随之改变
print(numpy_array)  # 输出: [100 200   3   4]
```

但是，需要注意的是，内存共享仅在 CPU 上的张量和数组之间有效。如果张量被移动到 GPU 上（使用 `.to('cuda')` 或 `.cuda()`），则内存共享不再适用，因为 NumPy 仅支持 CPU 操作。在这种情况下，你需要显式地进行数据拷贝来在 GPU 张量和 NumPy 数组之间转换数据。

#### 张量（tensors）移动到 GPU 上
将张量（tensors）移动到 GPU 上是深度学习中的一种常见操作，特别是在需要处理大型数据集或执行复杂的数学运算时。GPU（图形处理单元）由于其并行处理能力，相比于 CPU 可以大幅加速这些操作。以下是一些常见的情况，你可能需要将张量移动到 GPU 上：

1. **大规模矩阵运算**：深度学习中的很多操作，如卷积、矩阵乘法等，涉及大规模的矩阵运算。GPU 在处理这类运算时效率远高于 CPU。

2. **训练复杂的神经网络**：当训练大型或复杂的神经网络模型时，使用 GPU 可以显著减少训练时间。

3. **处理大型数据集**：在处理大规模数据集，特别是在图像或视频处理任务中时，GPU 可以提供更快的数据处理能力。

#### 如何将张量移动到 GPU

在 PyTorch 中，你可以使用 `.to()` 方法或 `.cuda()` 方法将张量移动到 GPU 上。这两种方法都需要你有一个支持 CUDA 的 GPU。`.to()` 方法更加通用，因为它允许你指定要移动到的设备。

首先，检查是否有可用的 CUDA 设备：

```python
import torch

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()

print("Is CUDA available:", cuda_available)
```

如果 CUDA 可用，你可以这样移动张量到 GPU：

```python
# 创建一个张量
tensor = torch.tensor([1, 2, 3, 4])

# 将张量移动到 GPU
if cuda_available:
    tensor = tensor.to('cuda')

print(tensor)
```

或者使用 `.cuda()` 方法：

```python
if cuda_available:
    tensor = tensor.cuda()

print(tensor)
```

#### 注意事项

- 在将张量移动到 GPU 后，所有的运算也需要在 GPU 上执行。这意味着你可能需要将你的模型和其他相关张量也移动到 GPU。

- 当你的数据在 GPU 上时，任何想要与 CPU 上的数据交互的操作（例如，打印输出或转换为 NumPy 数组）都需要先将数据移回 CPU。你可以使用 `.to('cpu')` 或 `.cpu()` 方法来实现。

- 使用 GPU 会占用显存。当处理大型数据或模型时，需要注意显存的使用情况，避免显存不足的错误。

- 在多 GPU 环境中，你可能需要明确指定使用哪个 GPU，这可以通过指定 CUDA 设备的索引来实现，例如 `tensor.to('cuda:0')` 或 `tensor.cuda(0)`。

#### `torch.tensor` 创建的张量默认是在 CPU 上


如果不特别指定，通过 `torch.tensor` 创建的张量默认是在 CPU 上的。你可以显式地指定设备来在 GPU 或 CPU 上创建张量。

以下是一个示例，其中：

1. 我们首先在 GPU 上直接创建一个张量（前提是你的机器上有可用的 CUDA 支持的 GPU）。
2. 然后，在 CPU 上创建另一个张量。
3. 将 CPU 上的张量移动到 GPU 上。
4. 在 GPU 上执行这两个张量的运算。
5. 输出运算结果。
6. 最后，将结果移回 CPU 并再次输出。

请注意，这个示例假设你的机器具有支持 CUDA 的 GPU。

```python
import torch

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()

# 在 GPU 上直接创建一个张量（如果 CUDA 可用）
if cuda_available:
    tensor_gpu = torch.tensor([1, 2, 3, 4], device='cuda')
else:
    print("CUDA not available. Example cannot be executed.")

# 在 CPU 上创建一个张量
tensor_cpu = torch.tensor([5, 6, 7, 8])

# 将 CPU 上的张量移动到 GPU（如果 CUDA 可用）
if cuda_available:
    tensor_cpu = tensor_cpu.to('cuda')

    # 在 GPU 上执行运算
    result_gpu = tensor_gpu + tensor_cpu

    # 在 GPU 上输出结果
    print("Result on GPU:", result_gpu)

    # 将结果移回 CPU 并输出
    result_cpu = result_gpu.to('cpu')
    print("Result back on CPU:", result_cpu)
```

在这个示例中，我们首先检查 CUDA 是否可用。如果可用，我们在 GPU 上创建 `tensor_gpu`，然后在 CPU 上创建 `tensor_cpu`，并将其移动到 GPU。接下来，我们在 GPU 上对这两个张量执行加法运算，并打印结果。最后，我们将结果移回 CPU 并再次打印。如果 CUDA 不可用，代码将输出相应的提示信息。

### 自动梯度求导

在 PyTorch 中，自动求导（Automatic Differentiation）是一个核心特性，它允许用户轻松地计算神经网络中的梯度。这在深度学习训练中是非常重要的，因为它涉及到根据损失函数对模型参数进行优化。

#### `requires_grad`

- `requires_grad` 是 PyTorch 张量的一个属性。当设置为 `True` 时，PyTorch 会开始跟踪在该张量上执行的所有操作，以便以后进行梯度计算。这对于训练过程中的参数是必要的，因为我们需要根据这些参数的梯度来更新它们。

  ```python
  import torch
  
  # 创建一个张量并设置 requires_grad=True 来跟踪它的梯度
  x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
  ```

#### `backward()`

- `backward()` 是计算梯度的主要方法。当在一个输出张量上调用此方法时（通常在损失函数之后），PyTorch 会自动计算梯度并将它们存储在相应张量的 `.grad` 属性中。

  ```python
  # 对一个简单函数进行操作并计算梯度
  y = x * x  # 任意操作
  z = y.sum()  # 求和得到一个标量
  z.backward()  # 计算梯度
  
  print(y)
  '''tensor([1., 4., 9.], grad_fn=<MulBackward0>)'''
  
  print(z)
  '''tensor(14., grad_fn=<SumBackward0>)'''
  
  print(x.grad)  #z对 x 的梯度将被计算并存储在这里
  '''tensor([2., 4., 6.])'''
  ```

在深度学习和自动微分（比如 PyTorch 中的自动梯度计算）的上下文中，**当我们计算一个标量对一个向量的梯度时，实际上是在计算标量输出相对于向量中每个独立变量的偏导数**。这是向量微积分的一个基本概念，通常用于优化问题，特别是在梯度下降算法中。
![image-20240206180803978](.\assets\image-20240206180803978.png)

#### 补充:梯度的基本概念

确实，"梯度"和"偏导数"这两个术语在某些上下文中可以互换使用，尤其是在涉及标量函数对向量求导的情况中。然而，它们在数学上有细微的区别：

1. **偏导数**：
   - 偏导数通常用于多变量函数。
   - 对于函数 $ f(x_1, x_2, ..., x_n) $，每个变量 $ x_i $ 的偏导数是当其他变量保持不变时，函数相对于该变量的变化率。
   - 偏导数是一个标量值，表示函数在某一点沿某个坐标轴的斜率。

2. **梯度**：
   - 梯度是一个向量，包含了一个多变量函数在某一点上所有偏导数的集合。
   - 对于函数 $ f(x_1, x_2, ..., x_n) $，其梯度是 $ \nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, ..., \frac{\partial f}{\partial x_n}\right] $。
   - 梯度向量指向函数增长最快的方向，其大小是在该方向上的最大斜率。

#### 标量对向量求梯度

给定一个标量函数 $ z = f(\mathbf{x}) $，其中 $ \mathbf{x} = [x_1, x_2, ..., x_n] $ 是一个向量，函数 $ f $ 对向量 $ \mathbf{x} $ 中的每个元素 $ x_i $ 的偏导数构成了梯度向量 $ \nabla f $=$ \frac{\partial z}{\partial x} $。梯度的每个分量 $ \frac{\partial z}{\partial x_i} $ 表示 $ z $ 相对于 $ x_i $ 的变化率。

例如，在你的代码中：

- 向量 $ \mathbf{x} = [1.0, 2.0, 3.0] $
- 函数 $ y = \mathbf{x} \cdot \mathbf{x} = x_1^2 + x_2^2 + x_3^2 $
- 标量 $ z = y.sum() = x_1^2 + x_2^2 + x_3^2 $

梯度 $ \nabla z $ 是：

$$ \nabla z = \begin{bmatrix} \frac{\partial z}{\partial x_1} \\ \frac{\partial z}{\partial x_2} \\ \frac{\partial z}{\partial x_3} \end{bmatrix} = \begin{bmatrix} 2x_1 \\ 2x_2 \\ 2x_3 \end{bmatrix} $$

当 $ \mathbf{x} = [1.0, 2.0, 3.0] $，梯度 $ \nabla z $ 就是 $ [2 \cdot 1.0, 2 \cdot 2.0, 2 \cdot 3.0] = [2.0, 4.0, 6.0] $。

#### 标量对矩阵求梯度

标量也可以对一个矩阵求偏导数，这在神经网络中调整权重矩阵时尤为常见。在这种情况下，你会得到一个与原始矩阵形状相同的梯度矩阵，其中每个元素是原始矩阵中相应元素的偏导数。

例如，如果有一个标量函数 $ z $ 依赖于一个矩阵 $ M $，那么 $ z $ 相对于 $ M $ 的梯度是一个矩阵，其元素定义为：$ \frac{\partial z}{\partial M_{ij}} $

其中 $ M_{ij} $ 是矩阵 $ M $ 中第 $ i $ 行第 $ j $ 列的元素。
![image-20240206180912973](.\assets\image-20240206180912973.png)

这种偏导数的计算在使用梯度下降进行机器学习模型训练时非常重要，它使得可以通过计算损失函数相对于模型参数（通常是权重矩阵）的梯度来优化这些参数。

```python
import torch

x=torch.ones(2,2,requires_grad=True)
y=x+2
print(y)
'''tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)'''

z=y*y*3
print(z)
'''tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>)'''

out=z.mean()
print(out)
'''tensor(27., grad_fn=<MeanBackward0>)'''

out.backward()
print(x.grad)#求out对x的梯度
'''tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])'''
```

![image-20240206193748618](.\assets\image-20240206193748618.png)

#### 标量对张量求梯度

在深度学习和自动微分的背景下，我们可以将梯度的概念推广到不同维度的张量。下面是一个概括：

1. **标量对标量求导数**：
   - 当我们有一个标量函数 $ f(x) $ 相对于另一个标量 $ x $ 求导时，结果是一个标量，这就是传统意义上的导数。

2. **标量对向量求梯度**：
   - 当我们有一个标量函数 $ f(\mathbf{x}) $ 相对于一个向量 $ \mathbf{x} = [x_1, x_2, ..., x_n] $ 求导时，结果是一个向量，即梯度。这个梯度向量包含了函数相对于每个分量的偏导数。

3. **标量对矩阵求梯度**：
   - 当我们有一个标量函数 $ f(\mathbf{M}) $ 相对于一个矩阵 $ \mathbf{M} $ 求导时，结果是一个与 $ \mathbf{M} $ 形状相同的矩阵。这个矩阵的每个元素是原始标量函数相对于矩阵中对应元素的偏导数。

4. **标量对高维张量求梯度**：
   - 对于高维张量（例如，张量的维度超过 2），情况类似。如果我们有一个标量函数 $ f(\mathbf{T}) $ 相对于一个高维张量 $ \mathbf{T} $ 求导，得到的梯度将是一个与 $ \mathbf{T} $ 形状相同的张量，其中包含了 $ f $ 相对于 $ \mathbf{T} $ 中每个元素的偏导数。

在所有这些情况中，基本的概念都是相同的：我们在寻找标量输出如何随着输入（无论是标量、向量、矩阵还是更高维张量）的变化而变化的速率。在实际的深度学习应用中，这些概念是通过自动微分库（如 PyTorch）实现的，这些库能够自动高效地计算这些梯度，这对于训练神经网络至关重要。

#### 标量对中间张量求梯度

在 之前的代码 中，`y` 是一个由 `x` 通过一系列操作（加法）得到的中间张量。由于 `y` 不是一个叶子节点（leaf tensor，即直接由用户创建的张量，对其 `requires_grad` 设置为 `True`），PyTorch 默认不会为其保留梯度。这是为了节省内存，因为在大多数情况下，用户通常只关心对最初叶子节点的梯度。

如果你确实需要计算并查看中间节点（如 `y`）的梯度，你可以使用 `.retain_grad()` 方法。这个方法会告诉 PyTorch 保留该节点在反向传播过程中的梯度。

在你的代码中，如果你想要查看 `y` 的梯度，你应该在进行反向传播之前对 `y` 调用 `.retain_grad()`。下面是修改后的代码示例：

```python
import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
y.retain_grad()  # 保留y的梯度
z = y * y * 3
out = z.mean()

out.backward()
print(x.grad)  # 求out对x的梯度
print(y.grad)  # 查看y的梯度
```

![image-20240206200632696](.\assets\image-20240206200632696.png)
这样，你就可以看到 `y` 在反向传播过程中计算出的梯度了。请注意，通常只有在特定的分析或调试需要时才这样做，因为保留所有中间节点的梯度会增加内存消耗。

#### 梯度累加

在 PyTorch 中，如果你想要计算相对于同一组变量的另一个不同标量函数的梯度，你可以再次执行反向传播（`.backward()`）过程。但需要注意的是，如果你不先清除现有的梯度，新的梯度会累加到现有梯度上。这是因为 PyTorch 默认会累积梯度，以便于在同一参数上进行多次反向传播。

例如，假设你已经对某个标量函数 `out1` 求了梯度，并想要对另一个标量函数 `out2` 求梯度。你可以这样做：

1. **清除现有梯度**：在进行第二次反向传播之前，首先清除之前累积的梯度，以避免梯度累加。这可以通过调用 `x.grad.zero_()` 来实现。

2. **计算新的梯度**：定义新的标量函数 `out2` 并对其调用 `.backward()`，来计算相对于同一变量的新梯度。

```python
import torch

# 创建张量 x 并设置 requires_grad=True 来跟踪它的梯度
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# 第一个标量函数 out1 = sum(x * x)
y1 = x * x
out1 = y1.sum()
# 反向传播计算 out1 对 x 的梯度
out1.backward()
print("梯度 after out1.backward():", x.grad)
'''tensor([[2., 4.],
        [6., 8.]])'''

# 清除现有的梯度
x.grad.zero_()

# 第二个标量函数 out2 = sum(x * x * x)
y2 = x * x * x
out2 = y2.sum()
# 反向传播计算 out2 对 x 的梯度
out2.backward()
print("梯度 after out2.backward():", x.grad)
'''tensor([[ 3., 12.],
        [27., 48.]])'''

#如果忘记x.grad.zero_()了，结果就是两次结果的累加
'''tensor([[ 5., 16.],
        [33., 56.]])'''
```

在上述代码中，`out1.backward()` 计算了 `out1` 相对于 `x` 的梯度，然后 `x.grad.zero_()` 清除了 `x` 的现有梯度，之后 `out2.backward()` 计算了 `out2` 相对于 `x` 的梯度。这样，`x.grad` 就会包含 `out2` 的梯度，而不是 `out1` 的梯度。
![image-20240206202743530](.\assets\image-20240206202743530.png)

总之，确保在计算新的梯度之前清除旧的梯度是非常重要的，除非你有意要累积梯度（例如，在某些优化算法中）。

#### grad_fn是什么

在 PyTorch 中，`grad_fn` 属性是一个非常重要的部分，它记录了张量的梯度是如何计算出来的。`grad_fn` 代表了“gradient function”，即梯度函数。这个属性是自动微分系统（autograd）的一部分，它在构建计算图（computational graph）时起到关键作用。

#### 什么时候会出现 `grad_fn`？

每当你对张量进行操作以创建一个新的张量时，新张量会有一个 `grad_fn` 属性。这个属性指向一个函数对象，这个函数对象表示了创建这个新张量的操作。

- **如果张量是直接创建的（例如使用 `torch.tensor`），那么 `grad_fn` 将是 `None`。**
- **如果 `requires_grad=True` 被设置在一个张量上，那么所有由这个张量经过运算得到的新张量都会有一个 `grad_fn`。**

#### `grad_fn` 的作用

`grad_fn` 存储了用于计算梯度的函数。**在反向传播（backward pass）时，PyTorch 通过这些函数自动计算梯度。**每个函数都知道如何计算其输出张量的梯度，以及如何将这些梯度传递给它的输入张量。

这种机制使得 PyTorch 能够轻松地计算复杂操作的梯度，而无需用户显式编写梯度计算代码。

#### 示例

考虑以下例子：

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(x.grad_fn)
#None

y = x * x
print(y.grad_fn)
#<MulBackward0 object at 0x00000240137AD190>

z = y.sum()
print(z.grad_fn)
#<SumBackward0 object at 0x00000240137AD190>
```

在这个例子中：

- `x` 是用户创建的张量，由于设置了 `requires_grad=True`，它可以有一个梯度，但 `grad_fn` 是 `None`，因为它不是由其他张量操作得来的。
- `y` 是 `x` 的平方，因此它有一个 `grad_fn`，这个 `grad_fn` 表示了乘法操作。
- `z` 是 `y` 的和，因此它也有一个 `grad_fn`，这个 `grad_fn` 表示了求和操作。

通过这种方式，PyTorch 构建了一个计算图，使得在进行梯度下降和模型优化时，可以自动计算梯度。
#### requires_grad_

在 PyTorch 中，即使在创建张量之后，你也可以改变其 `requires_grad` 属性。这可以通过直接设置张量的 `.requires_grad` 属性来实现。例如：

```python
import torch

# 创建一个张量，默认情况下 requires_grad=False
x = torch.tensor([1.0, 2.0, 3.0])

# 后续更改其 requires_grad 属性
x.requires_grad_(True)

# 现在 x 的 requires_grad 属性设置为 True
print(x.requires_grad)  # 输出: True
```

这里的 `x.requires_grad_(True)` 是一个就地操作（in-place operation），意味着它直接更改了原始张量 `x` 的 `requires_grad` 属性。**请注意函数名后面的下划线 `_`，在 PyTorch 中，这表示函数将对调用它的张量进行就地操作。**

这种方法在某些情况下非常有用，比如当你开始时不需要计算梯度，但在某个点之后需要开始对张量进行梯度跟踪时。通过这种方式，可以灵活地控制哪些计算应该包含在自动微分中。

#### `detach()`

- `detach()` 方法用于从当前计算图中分离出一个张量。分离后的张量不会在其上的操作中跟踪梯度。这对于阻止不必要的梯度计算非常有用，特别是在你只想进行前向传播而不关心反向传播时。

  ```python
  import torch
  
  x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
  y=x*x
  print(y)
  '''tensor([1., 4., 9.], grad_fn=<MulBackward0>)'''
  # 分离张量
  detached_x = x.detach()
  # 在 detached_x 上的操作不会被跟踪梯度
  y = detached_x * detached_x
  print(y)
  '''tensor([1., 4., 9.])'''
  ```

#### `torch.no_grad()`

- `torch.no_grad()` 是一个上下文管理器，用于临时关闭给定块中的所有张量的梯度计算。这在评估模型时非常有用，因为在模型评估阶段，我们通常不需要计算梯度。

  ```python
  x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
  # 在不计算梯度的情况下执行代码块
  with torch.no_grad():
      # 在这个块内，所有的计算都不会跟踪梯度
      y = x * x
      z = y.sum()
      print(y)
      '''tensor([1., 4., 9.])'''
      print(z)
      '''tensor(14.)'''
  	#沒有grad_fn
  ```

在神经网络训练过程中，这些工具和概念都非常重要，因为它们使得反向传播和梯度下降成为可能。通过使用这些工具，开发者能够有效地实现网络的自动更新和优化。

### torch.nn--neural networks

在PyTorch中，线性层的输出计算公式遵循基本的线性代数运算。给定输入矩阵$X$，线性层的权重$W$和偏置$b$，**线性层的输出$Y$可以通过下面的公式计算：**

$$ Y = XW^T + b $$

这里：
- **$X$ 是输入矩阵，假设其形状为$[N, \text{in\_features}]$，其中$N$是批量大小【样本数】，$\text{in\_features}$是每个输入样本的特征数量。**
- **$W$ 是权重矩阵，其形状为$[\text{out\_features}, \text{in\_features}]$，其中$\text{out\_features}$是输出特征的数量。**
- **$b$ 是偏置向量，形状为$[\text{out\_features}]$。**
- **$Y$ 是输出矩阵，形状为$[N, \text{out\_features}]$。**

**注意，$W^T$表示$W$的转置**。在PyTorch的实现中，这个转置是隐式进行的，意味着你不需要显式地转置$W$来计算$Y$；PyTorch会自动处理这一点。简而言之，这个公式表示的是每个输入向量与权重矩阵的线性组合，加上偏置，从而得到输出向量。

#### 线性层nn.Linear

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# 定义输入数据
input_data = torch.randn(2, 4)  # 2个样本，4个特征
print('Input Data:', input_data)
'''Input Data: tensor([[ 0.3367,  0.1288,  0.2345,  0.2303],
        [-1.1229, -0.1863,  2.2082, -0.6380]])'''

# nn.Linear: 全连接层/线性层
linear_layer = nn.Linear(4, 3)  # 输入特征维度为4，输出特征维度为3
print('Linear Layer:',linear_layer.weight.data, linear_layer.bias.data)
'''Linear Layer: tensor([[ 0.3854,  0.0739, -0.2334,  0.1274],
        [-0.2304, -0.0586, -0.2031,  0.3317],
        [-0.3947, -0.2305, -0.1412, -0.3006]]) tensor([ 0.0472, -0.4938,  0.4516])'''

output = linear_layer(input_data)
print("Linear Layer Output:")
print(output)
print(input_data@linear_layer.weight.t()+linear_layer.bias)
print()
'''Linear Layer Output:
tensor([[ 0.1611, -0.5502,  0.1866],
        [-0.9961, -0.8843,  0.8177]], grad_fn=<AddmmBackward0>)
tensor([[ 0.1611, -0.5502,  0.1866],
        [-0.9961, -0.8843,  0.8177]], grad_fn=<AddBackward0>)'''
```

#### 意义是什么?

为了更好理解这一个过程，我们把数字简化一点：

```python
import torch

# 创建输入数据矩阵 input_data
input_data = torch.tensor([[1, 2, 3, 1],
                           [0, 1, 2, 0]])  # 两行四列的矩阵

# 创建权重矩阵 weight
weight = torch.tensor([[1, 0, 1, 2],
                       [2, 1, 1, 0],
                       [0, 2, 2, 1]])  # 三行四列的矩阵

# 创建偏置项向量 bias
bias = torch.tensor([1, 2, 3])  # 三个数字的向量

# 计算输出数据 output_data
output_data = input_data @ weight.t() + bias
'''在 PyTorch 中，如果两个张量的形状不匹配，会进行自动广播（broadcasting）操作，使得它们的形状变得相同，然后进行相应的元素相加操作。

具体来说，在执行 input_data @ weight.t() + bias 操作时，PyTorch 会将 bias 向量沿着第一个维度（行）进行复制扩展，使其形状与 input_data 的形状匹配。然后，对应位置的元素进行相加。'''

print("Output Data:")
print(output_data)

'''tensor([[ 7,  9, 14],
        [ 3,  5,  9]])'''
```

**本质：**

![image-20240208133706466](.\assets\image-20240208133706466.png)

- **神经元**: 上述神经网络有4个输入神经元，3个输出神经元。
- **偏置项**: 这三个输出神经元的旁边有着3个偏置，对应偏置向量的三个元素。
- **（线性层）连接权重**: 对于每一个输出神经元都有4个输入神经元与它相连，对应着weight矩阵的一行，总共有12条边对应着weight矩阵的12个元素。
- **样本和特征**: Input矩阵的每一行代表着一个样本，一行当中的4个元素代表着一个样本的4个特征（比如说一只人的身高，体重，胸围，肺活量）。
- **矩阵运算和输入输出**: 当第一行进行矩阵运算的时候，代表着有4个数据输入了输入神经元，算出来3个输出，代表了有3个数据从输出神经元输出。

#### 收获：

现在我们应该能够比较好的理解神经网络中的**线性层**的概念了。

首先，我们需要提供数据。这个**数据通常是一个矩阵，矩阵的行数代表着我们采集了多少个样本，而矩阵的列数则代表着每个样本有多少特征**。(举个例子，如果我们采集了2个人的身高、体重、肺活量和胸围的数据，就可以对应着一个2行4列的矩阵。)

```python
input_data = torch.randn(2, 4)  # 2个样本，4个特征
```

接着，我们要构建一个全连接层，可以通过`nn.Linear`来构建，**我们有4个输入特征，而假设我们想要3个输出特征**(例如，在分类任务中，通常将输出层的特征数量设置为类别的数量)，**那么这个线性层的参数就应该是一个 4x3 的矩阵**。

```py
linear_layer = nn.Linear(4, 3) 
```

通过线性层（linear layer），我们可以将**数据(一个矩阵，每行代表一个样本)**传入神经网络，并得到一个**输出(也是一个矩阵，每行代表对应样本的映射)**。
```py
output = linear_layer(input_data)
```

在神经网络进行了这个变换之后，对于每一个样本，都会有一个对应输出。

#### 三维张量输入

当输入`X`是一个三维张量时【后面会遇到】，`nn.Linear`层仍然可以按照线性变换 $y = XW^T + b$ 的方式工作，但这里的处理方式会略有不同。

假设我们有一个三维张量 `X`，其形状为 `(N, L, D)`，其中：

- `N` 是批次大小（batch size），代表了数据集中独立元素的数量。
- `L` 是序列长度，对于非序列数据，这可以是任何其他逻辑维度。
- `D` 是每个元素的特征数量。

现在，我们想通过一个 `nn.Linear` 层，其有 `D` 个输入特征和 `M` 个输出特征，即 `in_features=D` 和 `out_features=M`。

在这种情况下，**`nn.Linear` 层会对输入张量的每一个 `(L, D)` 形状的切片独立地应用线性变换 $y = XW^T + b$，**其中 `W` 是权重矩阵，`b` 是偏置向量。最终的输出将会是一个 `(N, L, M)` 形状的张量，其中每个 `(L, M)` 形状的切片都是原始 `(L, D)` 切片经过变换后的结果。

![image-20240320155330142](./assets/image-20240320155330142.png)

```py
import torch
import torch.nn as nn

# 创建输入数据矩阵 input_data
input_data = torch.tensor([[[1, 2, 3, 1],
                           [0, 1, 2, 0]],

                           [[1, 2, 3, 1],
                           [0, 1, 2, 0]]], dtype=torch.float)  # 注意：输入数据需要是float类型，因为要进行权重赋值

# 创建权重矩阵 weight
weight = torch.tensor([[1, 0, 1, 2],
                       [2, 1, 1, 0],
                       [0, 2, 2, 1]], dtype=torch.float)  # 注意：权重需要是float类型

# 创建偏置项向量 bias
bias = torch.tensor([1, 2, 3], dtype=torch.float)  # 注意：偏置需要是float类型

# 定义一个权重相同的全连接层
linear_layer = nn.Linear(4, 3, bias=True)

# 将自定义的权重和偏置赋值给线性层
with torch.no_grad():  # 不跟踪这些操作的梯度，否则报错
    linear_layer.weight.copy_(weight)  # 使用copy_方法
    linear_layer.bias.copy_(bias)

# 使用自定义权重和偏置的线性层进行前向传播计算
output = linear_layer(input_data)

print("Linear Layer Output:")
print(output)

# 验证自定义权重和偏置的正确性
expected_output = input_data @ weight.t() + bias
print("Expected Output (Manual Calculation):")
print(expected_output)
'''tensor([[[ 7.,  9., 14.],
         [ 3.,  5.,  9.]],

        [[ 7.,  9., 14.],
         [ 3.,  5.,  9.]]])'''
```









#### 图像表示:

在RGB颜色模型中，每个像素的颜色由红色（R）、绿色（G）、蓝色（B）三种颜色的强度值组成。每种颜色的强度值通常是从0到255之间的整数，其中0表示没有颜色强度，255表示最大的颜色强度。

因此，RGB颜色模型中的三个数字分别代表了红色、绿色和蓝色的强度值。例如，(255, 0, 0) 表示纯红色，因为红色通道的强度值为255，而绿色和蓝色通道的强度值为0。同理，(0, 255, 0) 表示纯绿色，(0, 0, 255) 表示纯蓝色。

白色表示所有三种颜色通道的强度值都是最大的，即(255, 255, 255)。这是因为红、绿、蓝三种颜色同时具有最大的强度值，所以混合在一起会产生白色。

黑色则表示所有三种颜色通道的强度值都是最小的，即(0, 0, 0)。因为没有任何颜色通道的强度，所以呈现出来的是黑色。

一个 8x8 的 RGB 图像可以被表示为三个 8x8 矩阵的堆叠,在这种表示中，每个矩阵对应于图像的一个颜色通道：红色（R）、绿色（G）和蓝色（B）。因此，这个三维矩阵通常具有形状 `(3, 8, 8)`，其中：

1. 第一维度（大小为 3）表示颜色通道。
2. 第二和第三维度（大小均为 8）表示图像的高度和宽度。

具体来说：

- 第一个 8x8 矩阵代表红色通道，其中每个元素的值代表对应像素位置上红色成分的强度。
- 第二个 8x8 矩阵代表绿色通道，类似地，每个元素值表示绿色成分的强度。
- 第三个 8x8 矩阵代表蓝色通道，每个元素值表示蓝色成分的强度。

在这三个矩阵中，每个位置 `(i, j)` 的元素组合（来自三个不同的矩阵）共同决定了图像在 `(i, j)` 位置的颜色。例如，如果在位置 `(i, j)`，红色通道的强度很高，而绿色和蓝色通道的强度较低，则该位置的像素将呈现为红色。

![image-20240208143402662](.\assets\image-20240208143402662.png)

#### 卷积层nn.Conv2d

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 创建输入图像
input_image = torch.randn(2, 3, 4, 4)  # 模拟两张4*4的RGB图片
print("Input Image:", input_image)
'''tensor([[[[-0.5315,  1.9341,  0.1511,  1.5144],
          [ 1.7019,  0.9329, -1.3355,  0.6266],
          [-1.7312,  0.5713, -0.1252,  0.4666],
          [-3.2298, -0.4096,  1.2271,  1.5511]],

         [[ 0.8518, -1.0948,  0.1352,  1.0647],
          [ 1.3682, -1.2345, -0.1212,  0.5854],
          [-0.0889, -1.6021,  0.9367,  0.8532],
          [-1.0981,  0.9135,  2.1323, -0.0753]],

         [[-0.7717,  0.5126, -0.2049, -0.5287],
          [-0.1000, -1.0932, -1.0375,  0.6450],
          [ 0.4060, -0.1076,  0.4363, -0.1531],
          [ 0.4669, -1.2515, -2.0547,  1.5220]]],


        [[[-1.8227,  1.0861, -1.5873,  0.3975],
          [ 0.0675, -0.7460, -0.1973,  0.0588],
          [ 0.7497, -0.5529,  0.5076, -0.2007],
          [-1.0481, -0.3469, -1.2214, -0.3252]],

         [[-0.4622,  0.6078, -1.2077, -0.1657],
          [ 0.0217, -0.4707,  0.4299,  0.3756],
          [-0.4323, -0.4917, -1.3562,  0.2143],
          [ 0.8606,  0.7697, -1.8817,  0.5711]],

         [[-0.7198, -1.0438, -0.1037,  0.2093],
          [-1.1235,  0.0525,  1.0796,  0.9268],
          [-1.1657,  0.0490,  0.4068, -0.7868],
          [-1.6652,  0.1431,  0.1582,  0.2911]]]])'''
```

对于这一个模拟图片的理解：
![image-20240208175803651](.\assets\image-20240208175803651.png)

- 首先回顾一下之前的知识，**为什么这一个张量torch.randn(2, 3, 4, 4)会长这个样子？**
  最后一个数字4可以表示一个有4个元素的向量。
  倒数第二个数字4可以表示刚才的向量出现4次，构成一个矩阵。
  3表示刚才结构重复出现3次。
  2表示刚才的结构又出现了2次，最后又得到了这样的一个张量。

- **如何理解这一个图片？**
  首先这个**张量可以被分为前后两半，前一半是第一张图片，后一半是第二张图片。**
  而对于**前一半又可以分为三个切片，每个切片是一个矩阵，描述的是红色，绿色和蓝色的强度。**
  对于**每一个矩阵有4*4个元素表示的是，这个图片在16个像素点上的颜色。**
  举一个例子，第一个元素-0.5315表述的就是第一张图片第一个像素红色的强度。(只是一种模拟，RGB颜色应该在0~255之间)

  #### 定义卷积层
  ```python
  # 定义卷积层
  conv_layer = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1)
  print("Conv Layer:", conv_layer.weight.data.size() )
  print(conv_layer.weight.data)
  print(conv_layer.bias.data)
  '''
  Conv Layer: torch.Size([2, 3, 3, 3])
  tensor([[[[-0.1557,  0.0996,  0.0235],
            [ 0.1412, -0.0883, -0.1317],
            [-0.0318, -0.1100, -0.1404]],
  
           [[ 0.0009, -0.1890,  0.0354],
            [-0.1177, -0.0557, -0.0197],
            [-0.0698, -0.0495, -0.1255]],
  
           [[-0.1389,  0.1667,  0.0066],
            [ 0.1244, -0.0537, -0.1856],
            [ 0.1393, -0.1675,  0.0526]]],
  
  
          [[[-0.0617, -0.1856, -0.1634],
            [-0.0968, -0.1674,  0.1734],
            [-0.1853,  0.1329,  0.0713]],
  
           [[ 0.0616,  0.1537, -0.0095],
            [-0.0648, -0.1723,  0.1239],
            [-0.0507,  0.1083, -0.0630]],
  
           [[ 0.0680, -0.1302, -0.1113],
            [-0.1219, -0.0349,  0.1808],
            [ 0.0154,  0.0018, -0.0415]]]])
  tensor([0.1256, 0.1465])
  '''
  ```

  ![image-20240208180507220](.\assets\image-20240208180507220.png)

在这个特定的例子中，我们使用了一个 `nn.Conv2d` 卷积层，其配置为 `in_channels=3`, `out_channels=2`, 和 `kernel_size=3`。这意味着：

1. **输出通道数为2**：这表明我们有两个卷积核，**每个卷积核负责生成一个输出特征图**(学过cnn应该不难理解)。这两个卷积核可以理解为两个不同的特征提取器，每个提取器专注于捕捉输入数据中的不同特征。

2. **卷积核尺寸为3x3**：这表明每个卷积核在其工作面上是一个 3x3 的矩阵。这个矩阵定义了卷积核如何与其覆盖的局部区域进行交互，以提取特定的空间特征。

3. **输入通道数为3**：由于输入图像是RGB图像，拥有3个颜色通道，因此每个卷积核必须同时处理这三个通道。**这意味着每个卷积核实际上是一个 3x3x3 的立方体，其中“深度”为3**，对应于输入图像的三个颜色通道。

4. **偏置向量包含俩2元素**:在卷积神经网络中，**偏置（bias）的数量通常与卷积核（或过滤器）的数量相同**。每个卷积核有一个对应的偏置值。

综合以上点，我们有两个 3x3x3 的卷积核，每个核在三个输入通道上都有一个 3x3 的矩阵，这些核分别在输入图像上滑动以提取特征，并生成两个不同的输出特征图。

#### 进行卷积

```python
conv_output = conv_layer(input_image)
print("nn.Conv2d Output:")
print(conv_output.size())
print(conv_output)
'''
torch.Size([2, 2, 4, 4])
tensor([[[[ 0.5914, -0.8443,  0.3207,  0.3029],
          [ 0.6956, -0.2633, -0.2755,  0.0091],
          [ 1.0091,  0.0539, -0.4332,  0.3565],
          [-0.0718, -0.2377,  0.0800,  0.7624]],

         [[-0.2488, -0.2749, -1.1166, -0.2491],
          [ 0.5504,  0.3816,  0.2963,  0.2610],
          [-0.0412, -0.0039, -0.4768, -0.0611],
          [ 0.7517,  0.1665, -0.2231, -0.3370]]],


        [[[-0.2135,  0.4644, -0.2044,  0.5666],
          [-0.0925, -0.2376, -0.2448,  0.6950],
          [-0.0976,  0.7593, -1.6869,  1.1621],
          [ 0.2258,  0.2534, -0.2848, -0.0522]],

         [[-0.0054, -0.7709,  0.0086, -0.3171],
          [ 0.6791,  0.1246, -0.1360,  0.1951],
          [ 0.0818, -0.3583, -0.7911, -1.8213],
          [-0.1488,  0.4026, -0.3277,  0.3289]]]],
       grad_fn=<ConvolutionBackward0>)
'''
```


![image-20240208182052802](.\assets\image-20240208182052802.png)![image-20240208181629799](.\assets\image-20240208181629799.png)

在您的代码中，您使用了一个配置为 `in_channels=3`, `out_channels=2`, `kernel_size=3`, `stride=1`, `padding=1` 的卷积层 `nn.Conv2d`。

1. **输入图像尺寸和填充**：原始输入图像尺寸是 4x4（长和宽），并且有 3 个通道（RGB）。在卷积操作中，您对图像应用了大小为 1 的填充（padding），这将图像的每一边扩展 1 个像素，所以填充后的图像尺寸变为 6x6。

2. **卷积核尺寸和步长**：您使用了尺寸为 3x3 的卷积核，并且步长（stride）为 1。

3. **计算输出特征图尺寸**：根据公式 $\text{输出特征图尺寸} = \frac{\text{输入图像尺寸} - \text{卷积核尺寸} + 2 \times \text{填充}}{\text{步长}} + 1$，您计算得到每个特征图的尺寸仍然是 4x4。
   让我们分步骤理解这个公式：
   - **输入图像尺寸 + 填充**：首先，计算考虑填充后的输入图像尺寸。填充（padding）是在输入图像的边界上添加额外的像素（通常是零值），以保持特征图的尺寸或减少边界信息的损失。这一步是将原始输入图像尺寸加上两倍的填充值。
   - **减去卷积核尺寸**：然后，从填充后的图像尺寸中减去卷积核的尺寸。这样做是为了计算卷积核需要从一端移动多少长度才能移动到另一端，也就是移动的距离
   - **除以步长**：接下来，将上一步的结果除以步长（stride）。步长是卷积核在输入图像上移动的像素数。这一步计算卷积核需要移动多少次才能从一端移动到另一端，即确定了输出特征图的尺寸(每次移动都对应一个输出的格子)。
   - **加一**：最后，由于卷积核在初始位置时也会生成一个输出，因此在最终结果中需要加一。

4. **输出特征图的通道数**：由于您使用了两个卷积核（`out_channels=2`），因此对于每张输入图像，卷积层将输出两个图层。每个图层都是由相应的卷积核生成的。

5. **总输出**：由于输入包含两张图像，因此最终的输出将是一个包含两张图的张量，每张图有两层，每层尺寸是4*4。因此，输出张量的形状是 `[2, 2, 4, 4]`：
   - 第一个 `2` 表示有两张输出图像。
   - 第二个 `2` 表示每张图像有两层（由于有两个卷积核）。
   - 最后两个 `4` 表示每个特征图的尺寸是 4x4。

因此，最终输出是一个形状为 `[2, 2, 4, 4]` 的张量，这与预期相符。

#### *手动实现卷积

有点复杂，但是对于理解卷积神经网络很有帮助

```python
# 提取权重和偏差
weights = conv_layer.weight.data
bias = conv_layer.bias.data
# 手动实现卷积的函数
def manual_conv2d(input_img, weights, bias, stride=1, padding=1):
    # 添加填充
    input_padded = F.pad(input_img, (padding, padding, padding, padding), mode='constant', value=0)

    # 提取输入和权重的维度
    batch_size, in_channels, in_height, in_width = input_padded.shape
    out_channels, _, kernel_height, kernel_width = weights.shape

    # 计算输出维度
    out_height = (in_height - kernel_height) // stride + 1
    out_width = (in_width - kernel_width) // stride + 1

    # 初始化输出
    output = torch.zeros((batch_size, out_channels, out_height, out_width))

    # 执行卷积操作
    for i in range(batch_size):
        for j in range(out_channels):
            for k in range(out_height):
                for l in range(out_width):
                    h_start = k * stride
                    h_end = h_start + kernel_height
                    w_start = l * stride
                    w_end = w_start + kernel_width
                    output[i, j, k, l] = torch.sum(input_padded[i, :, h_start:h_end, w_start:w_end] * weights[j]) + bias[j]

    return output

# 应用手动卷积
manual_conv_output = manual_conv2d(input_image, weights, bias, stride=1, padding=1)

# 打印结果
print("Manual Conv2d Output:")
print(manual_conv_output.size())
print(manual_conv_output)
```

当然，我将逐一解释您提到的这些细节：

1. **`F.pad` 的用途**：
   `F.pad` 是 PyTorch 中的一个函数，用于对张量添加填充（padding）。填充通常用于保持卷积操作后的输出尺寸，或者防止边缘信息丢失。这个函数的基本用法是指定要在各个维度上添加的填充大小。

   例如，如果您有一个 2x2 的矩阵并希望在所有边缘添加 1 的填充，您将得到一个 4x4 的矩阵：
   ```python
   matrix = torch.tensor([[1, 2], [3, 4]])
   padded_matrix = F.pad(matrix, (1, 1, 1, 1), 'constant', 0)
   '''tensor([[0, 0, 0, 0],
           [0, 1, 2, 0],
           [0, 3, 4, 0],
           [0, 0, 0, 0]])'''
   ```
   这里 `(1, 1, 1, 1)` 表示在左、右、上、下各添加 1 的填充，`'constant'` 表示填充类型为常数，`0` 是填充的值。

2. **卷积核的长宽是否可以不同**：
   在卷积神经网络中，卷积核的高度和宽度确实可以不同，尽管在许多常见的应用中，它们是相等的。

2. **output[i, j, k, l]里面的i,j,k,l是什么**
   
   - `i`：这个变量代表的是第i张输出图像。在您的例子中，输出两张图像，所以 **`i` 会从 0 到 1 变化，表示输出的第一张和第二张图像**。
   
   - `j`：这个变量代表的是输出通道的索引，或者说是卷积核的索引。在您的例子中，有两个卷积核（因为 `out_channels=2`），**所以 `j` 会从 0 到 1 变化，分别对应输出图像的两个通道，也对应两个不同的卷积核**。
   
   - `k` 和 `l`：这两个变量代表的是在输出特征图上的空间位置**。`k` 代表输出图的行索引，而 `l` 代表列索引。**它们决定了卷积核在输入图像上的位置，这取决于步长和卷积核的大小。
   

![image-20240208193111241](.\assets\image-20240208193111241.png)

1. **核心卷积代码解释**：

   在手动实现的卷积操作中，最关键的部分是这一行代码：

   ```python
   output[i, j, k, l] = torch.sum(input_padded[i, :, h_start:h_end, w_start:w_end] * weights[j]) + bias[j]
   ```

   - `input_padded[i, :, h_start:h_end, w_start:w_end]`：这部分是对输入图像的选取。其中：

     - `i` 同上，**要输出第i张图像，就要用输入的第i张图像进行卷积。**
     - `:` 表示选择该图像的所有通道,也就是**小立方体深度应该是满的**。
     - 要输出第k行，第l列的数值，**卷积核应该在h_start:h_end这两行之间，在w_start:w_end这两列之间。**

     **这样我们就从若干张输入图像当中挑出来第i张，看作一个大立方体，并且从其中抠出来了一个小立方体，和第j个卷积核进行卷积。**

   - `weights[j]`：这是第 `j` 个卷积核的权重。

   - `torch.sum( ... )`：这个函数计算了输入图像的选定区域和卷积核权重之间的元素乘积的总和。

   - `+ bias[j]`：在求和后，加上对应卷积核的偏置值,**一个卷积核有一个偏置,用了第j个卷积核应该加上偏置bias[j]**。

   因此，循环中的这些索引结合起来，表示了在输入图像的批次 (`i`) 上，使用特定的卷积核 (`j`)，在特定的位置进行卷积操作，从而生成输出特征图的每个元素。
   ![image-20240208193442876](.\assets\image-20240208193442876.png)

不难验证用系统库的卷积**nn.Conv2d**与我们手写的卷积效果完全相同，也就是正确的。

#### 池化层

先介绍**最大池化**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

conv_output = torch.tensor([[[[ 0.5914, -0.8443,  0.3207,  0.3029],
          [ 0.6956, -0.2633, -0.2755,  0.0091],
          [ 1.0091,  0.0539, -0.4332,  0.3565],
          [-0.0718, -0.2377,  0.0800,  0.7624]],

         [[-0.2488, -0.2749, -1.1166, -0.2491],
          [ 0.5504,  0.3816,  0.2963,  0.2610],
          [-0.0412, -0.0039, -0.4768, -0.0611],
          [ 0.7517,  0.1665, -0.2231, -0.3370]]],


        [[[-0.2135,  0.4644, -0.2044,  0.5666],
          [-0.0925, -0.2376, -0.2448,  0.6950],
          [-0.0976,  0.7593, -1.6869,  1.1621],
          [ 0.2258,  0.2534, -0.2848, -0.0522]],

         [[-0.0054, -0.7709,  0.0086, -0.3171],
          [ 0.6791,  0.1246, -0.1360,  0.1951],
          [ 0.0818, -0.3583, -0.7911, -1.8213],
          [-0.1488,  0.4026, -0.3277,  0.3289]]]])
'''之前卷积层输出的结果，有两张图片，每张图片有两个通道，每个通道是4x4的矩阵'''

# nn.MaxPool2d: 2D 最大池化层
maxpool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
'''MaxPool2d层是用来执行最大池化操作的，它只是对输入数据进行池化操作，而不涉及任何可学习的参数。因此，在使用MaxPool2d层时，不应该期望像卷积层或线性层那样可以访问权重和偏置。'''

# 池化核大小为2x2，步长为2
output = maxpool_layer(conv_output)
print("MaxPooling Layer Output:")
print(output.size())
print(output)

```

![image-20240208230006017](.\assets\image-20240208230006017.png)
对于输入特征图的每个通道，`MaxPool2d`层使用指定大小的池化核（在本例中是2x2的池化核）在特征图上滑动，每次滑动的步幅由步长参数指定（在本例中是2)。**在每个2x2的窗口中，`MaxPool2d`层选择窗口内的最大值作为输出**，然后将这个最大值作为输出特征图的一个像素。

因此，对于输入的两张2通道的4x4图片，经过2x2大小的池化核和步长为2的池化操作后，每张图片的尺寸会减半，变为2x2大小。**由于每个通道上的操作是独立的，所以输出仍然是两个通道**。因此，输出是两张2通道的2x2的图片。

##### *手写池化操作

看懂之前的卷积操作，手动实现池化是不难的

```python
def manual_maxpool2d(input_tensor, kernel_size=2, stride=2):
    # 提取输入特征图的维度
    batch_size, channels, in_height, in_width = input_tensor.shape

    # 计算输出特征图的维度
    out_height = (in_height - kernel_size) // stride + 1
    out_width = (in_width - kernel_size) // stride + 1

    # 初始化输出特征图
    output = torch.zeros((batch_size, channels, out_height, out_width))

    # 执行最大池化操作
    for i in range(batch_size):
        for j in range(channels):
            for k in range(out_height):
                for l in range(out_width):
                    h_start = k * stride
                    h_end = h_start + kernel_size
                    w_start = l * stride
                    w_end = w_start + kernel_size

                    # 在当前池化窗口中提取最大值
                    window = input_tensor[i, j, h_start:h_end, w_start:w_end]
                    output[i, j, k, l] = torch.max(window)

    return output

# 测试手动实现的最大池化
manual_maxpool_output = manual_maxpool2d(conv_output, kernel_size=2, stride=2)

# 打印结果
print("Manual MaxPooling Output:")
print(manual_maxpool_output.size())
print(manual_maxpool_output)

```

![image-20240208230744873](.\assets\image-20240208230744873.png)

在手动实现的最大池化操作中，最关键的部分是以下两行代码：

```python
window = input_tensor[i, j, h_start:h_end, w_start:w_end]
output[i, j, k, l] = torch.max(window)
```

这里的每个变量代表的含义是：

- `output[i, j, k, l]`：这表示输出张量中的一个特定元素。其中：
  - **`i` 代表输出的第几张图像**。
  - **`j` 代表的是输出图像中的第几个通道。**
  - **`k` 和 `l` 分别代表输出特征图中的行和列索引。**

- `window = input_tensor[i, j, h_start:h_end, w_start:w_end]`：这行代码选取了输入特征图的一个局部区域（即“窗口”）。其中：
  - **`i` 和 `j` 指定了输入图像和通道。**
  - `h_start:h_end` 和 `w_start:w_end` 定义了在输入特征图上的池化窗口的位置。**行在h_start:h_end之间，列在w_start:w_end之间。**

- `torch.max(window)`：这个函数计算了选定窗口中的最大值。

关于变量范围的解释：

- **`i` 的范围是输入（出）图像的个数，即批次大小。**
- **`j` 的范围是输入（出）通道的数量。**
- **`k` 和 `l` 的范围是根据池化核的大小和步长计算得出的输出特征图的尺寸。**

综合来看，这段代码实现了最大池化的基本操作：在输入特征图的每个通道上，按照指定的步长和池化核大小提取局部区域的最大值，从而生成缩小尺寸但保留重要特征的输出特征图。

注意：在卷积神经网络中，**池化层的步长（stride）和池化窗口（kernel size）大小确实很常见地被设置为相同的值**（这里都是2），但这不是一个绝对的规则。这两个参数可以根据特定的应用和网络设计需求进行独立设置

**步长和池化窗口大小不同**：

- 如果步长小于池化窗口大小，那么池化窗口在滑动过程中会有重叠。这可能会导致在池化输出中保留更多的信息，但同时也可能增加计算量。

- 如果步长大于池化窗口大小，那么在连续的池化操作中会有一些输入像素被跳过，这可能导致信息的丢失。

  ![image-20240208232802212](.\assets\image-20240208232802212.png)

手写的池化操作结果和nn.MaxPool2d的结果完全相同，是正确的。
**其他说明：**（如果不熟悉cnn）

1. **图像样本数量的保持**：无论是卷积还是池化操作，输入图像的批次大小（即样本数量）通常与输出图像的批次大小相同。这意味着，如果您输入了 N 张图像（无论是单张图像还是一个包含 N 张图像的批次），您将得到 N 张经过操作的图像。这种设计确保了网络可以同时处理多个图像样本，而且每张输入图像都会对应一张输出图像。

2. **通道数的变化与保持**：
   - 对于**卷积操作**，输入和输出的通道数可以不同。输出通道数由卷积层的配置（特别是卷积核的数量）决定。例如，如果一个卷积层有 64 个卷积核，那么无论输入图像的通道数是多少，输出的通道数将是 64。
   - 对于**池化操作**，输入和输出的通道数是相同的。池化操作是在每个通道上独立进行的，不会改变通道数。这意味着，如果输入图像有 C 个通道，那么经过池化后的输出图像也将有 C 个通道。

#### 平均池化

和最大池化只是变了名字，从max变成avg(mean),不赘述。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

conv_output = torch.tensor([[[[ 0.5914, -0.8443,  0.3207,  0.3029],
          [ 0.6956, -0.2633, -0.2755,  0.0091],
          [ 1.0091,  0.0539, -0.4332,  0.3565],
          [-0.0718, -0.2377,  0.0800,  0.7624]],

         [[-0.2488, -0.2749, -1.1166, -0.2491],
          [ 0.5504,  0.3816,  0.2963,  0.2610],
          [-0.0412, -0.0039, -0.4768, -0.0611],
          [ 0.7517,  0.1665, -0.2231, -0.3370]]],


        [[[-0.2135,  0.4644, -0.2044,  0.5666],
          [-0.0925, -0.2376, -0.2448,  0.6950],
          [-0.0976,  0.7593, -1.6869,  1.1621],
          [ 0.2258,  0.2534, -0.2848, -0.0522]],

         [[-0.0054, -0.7709,  0.0086, -0.3171],
          [ 0.6791,  0.1246, -0.1360,  0.1951],
          [ 0.0818, -0.3583, -0.7911, -1.8213],
          [-0.1488,  0.4026, -0.3277,  0.3289]]]])
'''之前卷积层输出的结果，有两张图片，每张图片有两个通道，每个通道是4x4的矩阵'''

# nn.AvgPool2d: 2D 平均池化层
avgpool_layer = nn.AvgPool2d(kernel_size=2, stride=2)
'''AvgPool2d层是用来执行平均池化操作的，它只是对输入数据进行池化操作，而不涉及任何可学习的参数。因此，在使用AvgPool2d层时，不应该期望像卷积层或线性层那样可以访问权重和偏置。'''

# 池化核大小为2x2，步长为2
output = avgpool_layer(conv_output)
print("AvgPooling Layer Output:")
print(output.size())
print(output)


def manual_avgpool2d(input_tensor, kernel_size=2, stride=2):
    # 提取输入特征图的维度
    batch_size, channels, in_height, in_width = input_tensor.shape

    # 计算输出特征图的维度
    out_height = (in_height - kernel_size) // stride + 1
    out_width = (in_width - kernel_size) // stride + 1

    # 初始化输出特征图
    output = torch.zeros((batch_size, channels, out_height, out_width))

    # 执行平均池化操作
    for i in range(batch_size):
        for j in range(channels):
            for k in range(out_height):
                for l in range(out_width):
                    h_start = k * stride
                    h_end = h_start + kernel_size
                    w_start = l * stride
                    w_end = w_start + kernel_size

                    # 在当前池化窗口中提取平均值
                    window = input_tensor[i, j, h_start:h_end, w_start:w_end]
                    output[i, j, k, l] = torch.mean(window)

    return output

# 测试手动实现的平均池化
manual_avgpool_output = manual_avgpool2d(conv_output, kernel_size=2, stride=2)

# 打印结果
print("Manual AvgPooling Output:")
print(manual_avgpool_output.size())
print(manual_avgpool_output)

```

#### 非线性层

#### ReLU

**ReLU（Rectified Linear Unit）**：对于输入张量中的每个元素，如果元素大于0，则保持不变；如果元素小于等于0，则将其置为0。
$$  \text{ReLU}(x) = \max(0, x) $$

```python
import torch
import torch.nn as nn

input=torch.randn(2,3,3)
print("Input:",input)

relu_layer = nn.ReLU()
output = relu_layer(input)
print("ReLU Activation Output:")
print(output)

```

**手写ReLU:**

```python
# 手动实现ReLU激活函数
def custom_relu(input_tensor):
    # 大于0的元素保持不变，小于0的元素置为0
    return input_tensor * (input_tensor > 0).float()

# 调用手动实现的ReLU激活函数
output_custom = custom_relu(input_tensor)
print("Custom ReLU Activation Output:")
print(output_custom)
```

注：

```python
input_tensor = torch.tensor([[1, -2, 3], [0, 4, -5]])
print(input_tensor>0)
'''tensor([[ True, False,  True],
        [False,  True, False]]'''
```

#### Sigmoid

**Sigmoid**：对于输入张量中的每个元素，通过Sigmoid函数将元素映射到范围[0, 1]之间，常用于二分类问题的输出层。
$$\sigma(x) = \frac{1}{1 + e^{-x}} $$。

```python
import torch
import torch.nn as nn

input=torch.randn(2,2)
print("Input:",input)
print(input)

sigmoid_layer = nn.Sigmoid()
output = sigmoid_layer(input)
print("Sigmoid Activation Output:")
print(output)
```

**手写sigmoid**

```python
# 手动实现Sigmoid激活函数
def custom_sigmoid(input_tensor):
    return 1 / (1 + torch.exp(-input_tensor))

# 调用手动实现的Sigmoid激活函数
output_custom = custom_sigmoid(input)
print("Custom Sigmoid Activation Output:")
print(output_custom)
```

注:对一个矩阵进行Sigmoid激活函数操作时，PyTorch会自动对矩阵中的每个元素进行Sigmoid函数的计算，而不需要显式地编写循环来逐个处理每个元素。
```python
input_tensor = torch.tensor([[-100,0], [1,2]])
print(torch.exp(input_tensor))
'''tensor([[3.7835e-44, 1.0000e+00],
        [2.7183e+00, 7.3891e+00]])'''
```

#### Softmax

**Softmax**：对于输入张量中的每个元素，通过Softmax函数将元素映射到一个概率分布，常用于多分类问题的输出层。
$ \text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} $。

```python
import torch
import torch.nn as nn

input=torch.tensor([[-1,0,1],[0,1,2]]).float()#用整数可能报错，应该用浮点数

softmax_layer = nn.Softmax(dim=1)
# dim=1表示在第二个维度上进行Softmax计算，通常是在多分类问题的输出层使用
output = softmax_layer(input)
print("Softmax Activation Output:")
print(output)
'''tensor([[0.0900, 0.2447, 0.6652],
        [0.0900, 0.2447, 0.6652]])'''
print(nn.Softmax(dim=0)(input))
'''tensor([[0.2689, 0.2689, 0.2689],
        [0.7311, 0.7311, 0.7311]])'''
```

**手写softmax**

```python
def custom_softmax(input_tensor, dim=1):
    # 计算指数
    exp_input = torch.exp(input_tensor)
    # 沿着指定维度求和
    sum_exp = torch.sum(exp_input, dim=dim, keepdim=True)
    # 进行softmax计算
    softmax_output = exp_input / sum_exp
    return softmax_output

# 测试自定义softmax函数
output_custom = custom_softmax(input, dim=1)
print("Custom Softmax Output:")
print(output_custom)

output_custom = custom_softmax(input, dim=0)
print("Custom Softmax Output:")
print(output_custom)
```

![image-20240209161903308](.\assets\image-20240209161903308.png)

1. **torch.exp(input_tensor)**：PyTorch会自动将`exp()`函数应用于张量中的每个元素，而不需要手动编写循环。这就是PyTorch广播机制的一种应用。

2. **torch.sum(exp_input, dim=dim, keepdim=True)**：在这一步中，`torch.sum()`函数沿着指定的维度`dim`对`exp_input`张量进行求和。`keepdim=True`参数保持结果张量的维度，使得结果张量与原始张量具有相同的维度数，只是在指定维度上长度为1。

3. **exp_input / sum_exp**：在这一步中，`exp_input`张量被除以`sum_exp`张量。由于`sum_exp`的形状被扩展以匹配`exp_input`，所以这是一个逐元素的除法操作，其结果是一个形状与`exp_input`相同的张量。PyTorch的广播机制使得在不同形状的张量之间执行逐元素操作变得非常简单，而无需手动编写循环来处理不同形状的张量。

#### Tanh

**Tanh（Hyperbolic Tangent）**：对于输入张量中的每个元素，通过Tanh函数将元素映射到范围[-1, 1]之间。
 $$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$。

```python
import torch
import torch.nn as nn

input=torch.randn(2,3,3)
print("Input:",input)

tanh_layer = nn.Tanh()
output = tanh_layer(input)
print("Tanh Activation Output:")
print(output)
```

**手写Tanh**

```python
def custom_tanh(input_tensor):
    # 计算tanh函数
    tanh_output = (torch.exp(input_tensor) - torch.exp(-input_tensor)) / (torch.exp(input_tensor) + torch.exp(-input_tensor))
    return tanh_output

# 测试自定义tanh函数
output_custom = custom_tanh(input)
print("Custom Tanh Output:")
print(output_custom)
```

很简单，逐元素进行即可


#### Leaky ReLU

**Leaky ReLU**：与ReLU类似，但当输入为负数时，Leaky ReLU不会将其完全置为0，而是乘以一个小的斜率，以避免“神经元死亡”问题。
$
\text{ELU}(x) = 
\begin{cases} 
x & \text{if } x \geq 0 \\
\alpha(e^x - 1) & \text{if } x < 0 
\end{cases}
$

```python
import torch
import torch.nn as nn

input=torch.randn(2,3,3)
print("Input:",input)


leaky_relu_layer = nn.LeakyReLU(negative_slope=0.01)  # 可以自定义负斜率，通常为小于1的正数
output = leaky_relu_layer(input)
print("Leaky ReLU Activation Output:")
print(output)
```

**手写Leaky ReLU**

```python
def custom_leaky_relu(input_tensor, negative_slope=0.01):
    # 对于每个元素，如果大于等于0，保持不变；如果小于0，乘以负斜率
    leaky_relu_output = torch.where(input_tensor >= 0, input_tensor, input_tensor * negative_slope)
    return leaky_relu_output

# 测试自定义Leaky ReLU函数
output_custom = custom_leaky_relu(input)
print("Custom Leaky ReLU Output:")
print(output_custom)

```

注:

`torch.where()` 函数是 PyTorch 中的一个张量操作函数，用于根据条件选择两个张量中的元素。它的用法如下：

```python
torch.where(condition, x, y)
```

其中，`condition` 是一个布尔类型的张量，用于指定选择的条件。如果 `condition` 中的元素为 `True`，则选择 `x` 中对应位置的元素；如果 `condition` 中的元素为 `False`，则选择 `y` 中对应位置的元素。

举个简单的例子：

```python
import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
condition = torch.tensor([True, False, True])

result = torch.where(condition, x, y)
print(result)  # 输出: tensor([1, 5, 3])
```

在这个例子中，`condition` 为 `[True, False, True]`，所以在对应位置上选择了 `x` 中的元素 `[1, , 3]`，y中的元素`[,5,]`。

#### ELU

**ELU（Exponential Linear Unit）**：对于输入张量中的每个元素，如果元素大于0，则保持不变；如果元素小于等于0，则通过指数函数进行平滑处理。
$
\text{ELU}(x) = 
\begin{cases} 
x & \text{if } x \geq 0 \\
\alpha(e^x - 1) & \text{if } x < 0 
\end{cases}
$

```python
import torch
import torch.nn as nn

input=torch.randn(2,3,3)
print("Input:",input)

elu_layer = nn.ELU(alpha=1.0)  # 可以自定义alpha参数，通常为1.0
output = elu_layer(input)
print("ELU Activation Output:")
print(output)

```

**手写ELU**

```python
def custom_elu(input_tensor, alpha=1.0):
    return torch.where(input_tensor >= 0, input_tensor, alpha * (torch.exp(input_tensor) - 1))

# 测试自定义 ELU 函数
output_custom = custom_elu(input)
print("Custom ELU Output:")
print(output_custom)

```

代码是类似的：构建三个矩阵，第一个矩阵是x，第二个是$\alpha(e^x - 1)$,第三个是`x>0`与否，根据`x>0`选择元素。

```python
print(input>=0)#之前说过，这是一个bool型张量
```

这些激活函数在深度学习中经常被使用，并且在PyTorch中都有相应的实现。通过在神经网络中选择合适的激活函数，可以提高模型的表达能力和性能。
#### 归一化BatchNorm2d

![image-20240214132546918](.\assets\image-20240214132546918.png)

```python
import torch
import torch.nn as nn

# 定义手动实现的批量归一化函数
def manual_batchnorm(input_tensor, eps=1e-5):
    # 计算每个通道的均值和方差
    mean = input_tensor.mean([0, 2, 3], keepdim=True)
    var = input_tensor.var([0, 2, 3], keepdim=True, unbiased=False)
    print(mean)
    print(var)

    # 归一化
    normalized_tensor = (input_tensor - mean) / torch.sqrt(var + eps)
    return normalized_tensor

# 创建输入张量
input_tensor = torch.randn(2, 3, 4, 4)

# 使用 PyTorch 内置的批量归一化
batchnorm = nn.BatchNorm2d(3, eps=1e-5, momentum=0, affine=False)
output = batchnorm(input_tensor)

# 使用手动实现的批量归一化
manual_output = manual_batchnorm(input_tensor)

# 比较两种方法的输出
print("PyTorch BatchNorm2d Output:")
print(output)
print("\nManual BatchNorm Output:")
print(manual_output)

```

注:目标是对每个通道的数据进行独立的归一化处理，为了做到这一点，我们需要在除了通道维度（第二维，索引为 1）之外的所有维度上进行计算。
加上一个很小的数 `eps` 以避免除以零的情况。



### pytorch构建神经网络：线性回归

![image-20240211180135422](.\assets\image-20240211180135422.png)

线性回归是一种预测连续值的监督学习算法，多变量线性回归意味着模型的输入包含多个特征（变量）。以下是一步一步创建和训练一个简单的多变量线性回归模型的过程：

#### 准备数据

首先，我们需要创建一些合成数据来模拟我们的问题。假设我们有一个模型，它根据两个特征（x1 和 x2）来预测目标值 y。

```python
import torch
import torch.nn as nn
from torch.optim import SGD

# 假设的特征和权重
true_weights = torch.tensor([2.0, -3.5])
true_bias = torch.tensor([5.0])

# 创建一些合成数据
x_data = torch.randn(100, 2)  # 100个样本，每个样本2个特征
print('x_data',x_data)
y_data = x_data @ true_weights + true_bias  # @表示矩阵乘法
print(y_data)

# 在y_data中添加一些随机数
random_noise = torch.randn(y_data.shape)   # 添加正态分布的随机数
y_data += random_noise
```

在模拟线性回归数据时添加随机噪声是为了模拟真实世界数据中的不确定性和测量误差。在现实世界的数据中，很少有情况是完全线性的或完全无误差的。
即使添加了噪声，只要噪声不是系统性的偏差，并且模型是正确指定的（即，模型形式能够捕捉数据的真实关系），线性回归模型通常仍然能够估计出接近真实的权重和偏置参数，展示出其对数据生成过程的良好拟合。
![image-20240208120524170](.\assets\image-20240208120524170.png)

**我们的任务:已知x_data、y_data，求weight和bias**

#### 手动实现梯度下降

要使用手动梯度下降的方法进行线性回归，我们需要遵循以下步骤：

1. **初始化权重和偏置**：从一些随机值开始。
2. **选择损失函数**：通常用于线性回归的是均方误差（MSE）。
3. **设置学习率**：这是梯度下降中的一个重要超参数。
4. 迭代更新权重
   ![image-20240211115526202](.\assets\image-20240211115526202.png)
   - 计算预测值 `y_pred`。
   - 计算损失 `loss`。
   - 计算损失对权重和偏置的梯度。
   - 更新权重和偏置。

```python
# 初始化权重和偏置
weights = torch.randn(2, requires_grad=True)
bias = torch.randn(1, requires_grad=True)

# 设置学习率
learning_rate = 0.01

# 迭代次数
iterations = 1000

# 损失函数 - 均方误差 - Mean Squared Error Loss
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

# 执行梯度下降
for _ in range(iterations):
    # 计算预测值
    y_pred = x_data @ weights + bias

    # 计算损失
    loss = mse_loss(y_pred, y_data)

    # 计算梯度
    loss.backward()

    # 更新权重和偏置，使用 torch.no_grad() 来暂停梯度追踪
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        bias -= learning_rate * bias.grad

        # 清零梯度,否则梯度默认会叠加
        weights.grad.zero_()
        bias.grad.zero_()

# 打印结果
print(f"Estimated weights: {weights}")
print(f"Estimated bias: {bias}")
'''Estimated weights: tensor([ 1.9858, -3.5081], requires_grad=True)
Estimated bias: tensor([5.1391], requires_grad=True)'''
```

![image-20240211113845216](.\assets\image-20240211113845216.png)

当然，我可以用数学符号来解释这两个公式：

1. **预测值的计算公式**：

   假设我们有一个线性模型，其预测值$\hat{y} $ 是由输入数据$X $、权重$w $ 和偏置$b $ 通过线性关系计算得出的。数学上，这可以表示为：

   $$\hat{y} = Xw + b $$

   其中，
   - $ \hat{y} $ 是预测值。
   - $ X $ 是输入数据的矩阵。
   - $ w $ 是模型的权重向量。
   - $ b $ 是模型的偏置项。

   在这个公式中，矩阵$X $ 与向量$w $ 进行矩阵乘法，然后加上偏置$b $ 以得到预测值$\hat{y} $。

2. **均方误差损失函数（MSE Loss）的计算公式**：

   均方误差损失函数用于衡量预测值$\hat{y} $ 与真实值$y $ 之间的差异。其数学公式为：

   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 $$

   其中，
   - $ \hat{y}_i $ 是第$i $ 个样本的预测值。
   - $ y_i $ 是第$i $ 个样本的真实值。
   - $ n $ 是样本数量。

   这个公式计算了预测值和真实值之差的平方，然后对所有样本求平均，得到整体预测误差的均方值。

在这两个公式中，预测公式是线性模型的基础，而MSE损失函数是衡量预测准确性的常用方法，特别是在回归问题中。

#### 手动计算梯度

```python
import torch

# 假设的特征和权重
true_weights = torch.tensor([2.0, -3.5])
true_bias = torch.tensor([5.0])

# 创建一些合成数据
x_data = torch.randn(100, 2)  # 100个样本，每个样本2个特征
print('x_data',x_data)
y_data = x_data @ true_weights + true_bias  # @表示矩阵乘法
print(y_data)

# 在y_data中添加一些随机数
random_noise = torch.randn(y_data.shape)   # 添加正态分布的随机数
y_data += random_noise

# 重新初始化权重和偏置
weights = torch.randn(2, requires_grad=False)
bias = torch.randn(1, requires_grad=False)

# 损失函数 - 均方误差
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

# 设置学习率
learning_rate = 0.01

# 迭代次数
iterations = 1000

# 执行手动梯度计算和更新
for _ in range(iterations):
    # 计算预测值
    y_pred = x_data @ weights + bias

    # 计算损失
    loss = mse_loss(y_pred, y_data)

    # 手动计算梯度
    grad_w = (2.0 / x_data.shape[0]) * (x_data.t() @ (y_pred - y_data))
    grad_b = (2.0 / x_data.shape[0]) * torch.sum(y_pred - y_data)

    # 更新权重和偏置
    weights -= learning_rate * grad_w
    bias -= learning_rate * grad_b

# 打印结果
print(f"Estimated weights: {weights}")
print(f"Estimated bias: {bias}")
```

![image-20240211114605931](.\assets\image-20240211114605931.png)

当使用均方误差（Mean Squared Error, MSE）作为损失函数时，对于线性模型 $ y = wx + b $（其中 $ w $ 是权重，$ b $ 是偏置），损失函数关于权重 $ w $ 和偏置 $ b $ 的梯度可以通过微积分的方法得到。这些梯度的公式如下：

1. 权重的梯度 $ \nabla_w L $ 是损失 $ L $ 对 $ w $ 的偏导数，给出了 $ w $ 在减少损失方面应该如何改变。对于 MSE 损失，这个梯度是 $ \frac{2}{n} \times X^T \times (y_{pred} - y_{true}) $，其中 $ X^T $ 是输入数据的转置，$ y_{pred} $ 是模型预测，$ y_{true} $ 是真实标签。

2. 偏置的梯度 $ \nabla_b L $ 是损失 $ L $ 对 $ b $ 的偏导数，给出了 $ b $ 在减少损失方面应该如何改变。其梯度是 $ \frac{2}{n} \times \sum(y_{pred} - y_{true}) $。

这些梯度计算是基于数学推导得到的，通常在机器学习、深度学习和统计学的基础教材或课程中会进行介绍。

#### 正规方程法

要使用正规方程（Normal Equation）求解线性回归的权重 $ w $ 和偏置 $ b $，可以应用以下公式：

$$ w = (X^TX)^{-1}X^Ty $$

在这个公式中，$ X $ 是设计矩阵（即包含输入特征的矩阵，每个样本一行），$ y $ 是目标值向量。在我们的案例中，因为我们还需要计算偏置项 $ b $，所以我们需要向设计矩阵 $ X $ 添加一列全为1的列，以便包括偏置项。

```python
import torch

# 假设的特征和权重
true_weights = torch.tensor([2.0, -3.5])
true_bias = torch.tensor([5.0])

# 创建一些合成数据
x_data = torch.randn(100, 2)  # 100个样本，每个样本2个特征
y_data = x_data @ true_weights + true_bias  # @表示矩阵乘法

# 在y_data中添加一些随机数
random_noise = torch.randn(y_data.shape)   # 添加正态分布的随机数
y_data += random_noise

# 向x_data添加一列1以包括偏置项
X_with_bias = torch.cat([x_data, torch.ones(x_data.shape[0], 1)], dim=1)

# 使用正规方程求解权重和偏置
w_with_bias = torch.inverse(X_with_bias.t() @ X_with_bias) @ X_with_bias.t() @ y_data

# 提取权重和偏置
estimated_weights = w_with_bias[:-1]
estimated_bias = w_with_bias[-1]

# 打印结果
print(f"Estimated weights: {estimated_weights}")
print(f"Estimated bias: {estimated_bias}")

```

通过应用正规方程，我们得到了估计的权重和偏置值。这种方法直接使用数学公式计算出最优的权重和偏置，而不需要迭代的梯度下降过程。

最终得到的估计权重和偏置为：

- Estimated weights: tensor([ 1.9375, -3.5818])
- Estimated bias: 5.003337860107422

这些估计值非常接近我们的模拟数据中使用的真实权重（2.0, -3.5）和真实偏置（5.0）。这说明正规方程成功地从数据中恢复出了近似的线性关系。这种方法在数据集不是特别大且特征数量不是非常多的情况下非常有效。

#### *正规方程的证明：几何法

正规方程 $ w = (X^TX)^{-1}X^Ty $ 实际上可以从几何角度通过考虑目标向量 $ y $ 在由 $ X $ 的列向量构成的空间内的投影来推导。这个过程基于最小二乘法的原理，即找到一个解使得预测值和真实值之间的差异（误差）最小。

![image-20240211123355498](.\assets\image-20240211123355498.png)

![image-20240211123453753](.\assets\image-20240211123453753.png)

#### *正规方程的证明：代数法

![image-20240211145358977](.\assets\image-20240211145358977.png)

![image-20240211145433232](.\assets\image-20240211145433232.png)

1. **定义目标**：我们希望找到一个权重向量 $ w $，使得 $ Xw $ 尽可能接近 $ y $。这里，$ X $ 是设计矩阵，其列代表不同的特征，$ y $ 是目标向量。

2. **误差向量**：定义误差向量 $ e = y - Xw $。我们的目标是最小化这个误差向量的长度，即最小化 $ e^Te $。

3. **最小化误差**：要最小化 $ e^Te $，即最小化 $ (y - Xw)^T(y - Xw) $。展开这个表达式，我们得到 $ y^Ty - y^TXw - w^TX^Ty + w^TX^TXw $

   - 展开的细节

     1. **展开转置**：首先，我们应用转置运算的性质。如果有两个向量或矩阵 $A$ 和 $B$，那么 $(A - B)^T$ 等于 $A^T - B^T$。所以，$(y - Xw)^T$ 可以写为 $y^T - (Xw)^T$。

     2. **进一步转置**：接着，应用转置运算的另一个性质，即 $(AB)^T = B^TA^T$。这意味着 $(Xw)^T$ 等于 $w^TX^T$。

     3. **将转置应用到整个表达式**：因此，原始表达式变为 $y^T - w^TX^T$ 乘以 $y - Xw$。

     4. **应用分配律**：接下来，我们使用矩阵乘法的分配律将这个乘法展开。这就像展开普通的多项式一样，即 $A(B - C) = AB - AC$。所以我们得到：

        $$ (y^T - w^TX^T)(y - Xw) = y^Ty - y^TXw - w^TX^Ty + w^TX^TXw $$

     在这里：
        - $y^Ty$ 是一个标量，表示向量 $y$ 的元素平方和。
        - $y^TXw$ 和 $w^TX^Ty$ 是等价的，因为它们都表示相同的内积（y和Xw的内积）。它们也是标量。
        - $w^TX^TXw$ 是向量 $Xw$ 的元素平方和，也是一个标量。


1. **求解最小值**：要找到 $ w $ 使得上述表达式最小，我们对 $ w $ 求导并设其为零。这给出 $ -2X^Ty + 2X^TXw = 0 $。

   - 求导的细节

   - 当我们对表达式 $ y^Ty - y^TXw - w^TX^Ty + w^TX^TXw $ 中的 $ w $ 求导时，需要考虑到 $ w $ 是一个向量。这里的求导涉及到向量微积分的知识。我们逐项对 $ w $ 求导：

     1. **对 $ y^Ty $ 求导**：这一项与 $ w $ 无关，所以它的导数为零。

     2. **对 $ -y^TXw $ 求导**：这一项是一个关于 $ w $ 的线性项。它的导数是 $ -X^Ty $。这里我们应用了向量微积分中的规则：如果 $ y = Xw $，那么 $ \frac{\partial y}{\partial w} = X^T $。

     3. **对 $ -w^TX^Ty $ 求导**：由于矩阵乘法的性质，这一项与上一项相同，其导数也是 $ -X^Ty $。

     4. **对 $ w^TX^TXw $ 求导**：这是一个关于 $ w $ 的二次项。根据向量微积分的规则，$ \frac{\partial (w^T A w)}{\partial w} = 2Aw $（其中 $ A $ 是一个对称矩阵）。因此，这一项的导数是 $ 2X^TXw $。

     将所有这些部分组合起来，我们得到：

     $$ \frac{\partial}{\partial w} (y^Ty - y^TXw - w^TX^Ty + w^TX^TXw) = 0 - X^Ty - X^Ty + 2X^TXw = -2X^Ty + 2X^TXw $$

     所以，最终的导数是 $ -2X^Ty + 2X^TXw $。在最小化损失函数的上下文中，将这个导数设为零可以帮助我们找到最优的 $ w $ 值。

2. **解方程**：简化上述方程，我们得到 $ X^TXw = X^Ty $。如果 $ X^TX $ 是可逆的，我们可以两边同时乘以 $ (X^TX)^{-1} $，得到 $ w = (X^TX)^{-1}X^Ty $。

这就是正规方程的推导过程。它基于最小化误差向量的二范数，这也就是为什么这种方法被称为最小二乘法。这种方法在 $ X^TX $ 可逆的情况下特别有效，但如果 $ X $ 的列之间高度相关（即存在多重共线性）或者 $ X $ 的列数远大于样本数（即过度确定的情况），则 $ X^TX $ 可能不可逆或者非常接近奇异，这种情况下需要考虑其他方法，如岭回归（Ridge Regression）。

#### pytorch方法

#### 步骤·1:准备数据

**如果不熟悉python的类和继承，后面的pytorch语法可能比较难懂**

回顾一下python当中类、类的成员、构造函数、继承、重写的概念
```python
# 定义一个父类Animal
class Animal:
    def __init__(self, name):#在类的方法中，第一个参数通常都是self，它表示类的实例本身，
        self.name = name

    def speak(self):
        print(f"{self.name} makes a sound")

# 定义一个子类Dog，继承自Animal
class Dog(Animal):
    def __init__(self, name, breed):
        # 调用父类的初始化方法，并传入name参数
        super().__init__(name)#等价于self.name = name
        self.breed = breed

    # 子类可以重写父类的方法
    def speak(self):
        print(f"{self.name} barks")

    def fetch(self):
        print(f"{self.name} fetches the ball")

# 创建一个Animal类的实例
animal = Animal("Generic Animal")
animal.speak()  # 输出: Generic Animal makes a sound

# 创建一个Dog类的实例
dog = Dog("Buddy", "Golden Retriever")
dog.speak()    # 输出: Buddy barks
dog.fetch()    # 输出: Buddy fetches the ball

```

1. **类（Class）**：类是面向对象编程的基本组成单元。在python中，通过**关键字 `class` 来定义一个类。**

2. **构造函数（Constructor）**：在 Python 中，**构造函数的方法名为 `__init__()`**，通过在类中定义 `__init__()` 方法来创建构造函数。构造函数可以接受参数，用于初始化对象的属性。

3. **类的成员**：类的成员包括属性（属性是类的数据成员）和方法（方法是类的函数成员）。在上面的代码中，**`name` 和 `breed` 是类的属性，`speak()` 和 `fetch()` 是类的方法。**

4. **继承（Inheritance）**：继承是面向对象编程中一种重要的概念，它允许一个类（子类）继承另一个类（父类）的属性和方法。在 Python 中，**通过在定义子类时在类名后面添加括**号，并指定父类的名称来实现继承。**在子类中可以使用 `super()` 函数来调用父类的方法。**

5. **重写（Override）**：重写是指**子类重新定义或覆盖父类中的方法**，以实现自己的功能或行为。在上面的代码中，**`Dog` 类重写了父类 `Animal` 中的 `speak()` 方法**，以实现不同的行为。

在上面的例子中，**`super().__init__(name)`可以替换为`self.name = name`达到相同的效果**，但是这样会导致子类无法调用父类的初始化方法，从而破坏了继承关系，因此不推荐这样做。正确的做法是在子类的构造函数中使用`super().__init__(name)`来调用父类的初始化方法，以确保父类的初始化逻辑也被执行。

**`super()`函数的括号内是可以传入参数的**，这些参数用于指定在哪个类的上下文中执行方法。举个例子：

```python
class Parent:
    def __init__(self, name):
        self.name = name

class Child(Parent):
    def __init__(self, name, age):
        super(Child, self).__init__(name)  # 在子类中调用父类的初始化方法，并传入name参数
        self.age = age

child = Child("Alice", 5)
print(child.name)  # 输出: Alice
print(child.age)   # 输出: 5
```

在绝大多数情况下，`super().__init__(name)`与`super(Child, self).__init__(name)`是等价的。但是，在多重继承的情况下，这两者可能会产生不同的结果。

#### 步骤 2: 定义模型

**我们将使用 PyTorch 的 `nn.Module` 类来定义我们的模型**。对于线性回归，我们可以使用 PyTorch 提供的 `nn.Linear` 层。
在 PyTorch 中，`LinearRegressionModel` 是您自定义的类的**名字，并不是系统库的**。您可以根据您的需求给这个类命名，但最好选择一个描述性的名称，以便清楚地表明类的功能。
当您定义一个类并在括号里写 `nn.Module` 时，您的类 `LinearRegressionModel` **成为 `nn.Module` 的子类**。这意味着您的类继承了 `nn.Module` 的所有功能。

在子类的`__init__()`方法中，**通过`super().__init__()`调用父类的`__init__()`方法，来初始化**继承自父类的成员变量和方法。

而在`__init__()`方法中，通过`self.linear = nn.Linear(input_size, 1)`这一语句，创建了一个`nn.Linear`对象，并将其**赋值给了子类的成员变量`self.linear`**。这样，子类就拥有了一个线性层对象，可以在前向传播中使用。

```python
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        # 定义模型的层
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # 前向传播函数-
        return self.linear(x)

# 实例化模型
model = LinearRegressionModel(input_size=2)
```

`LinearRegressionModel` 类继承自 `nn.Module`，并且**重写了 `forward()` 方法**。`model.forward()`方法会在**调用`model()`时自动触发**。

事实上，之前

```python
output_data=nn.Linear(4, 3)(input_data)#会自动调用nn.Linear类的forward()方法
```

就会调用forward()方法。

由于我们的任务是构建`y=x1*w1+x2*w2+b`,也就是2输入特征，1输出特征，所以线性层参数应该是`nn.Linear(2,1)`。

#### 步骤 3: 定义损失函数和优化器

接下来，我们需要定义一个损失函数和一个优化器，用于训练模型。

```python
# 均方误差损失函数
loss_function = nn.MSELoss()

# 随机梯度下降优化器
optimizer = SGD(model.parameters(), lr=0.01)
```

**优化器是**机器学习和深度学习中用于**最小化**（或最大化）**损失函数**或目标函数**的方法**。

1. **Stochastic Gradient Descent (SGD)随机梯度下降**：
   - SGD是最基本的优化器之一。它对每个训练样本或小批量样本计算梯度，并相应地更新模型参数。
   - SGD通常伴随着一个学习率参数，有时还会使用动量（momentum）来加速训练。
2. **Adam (Adaptive Moment Estimation)**：略
3. **RMSprop (Root Mean Square Propagation)**：略

`optimizer = SGD(model.parameters(), lr=0.01)` 表示使用随机梯度下降（Stochastic Gradient Descent，简称 SGD）作为优化器来更新模型的参数**。这里的 `model.parameters()` 是指模型中所有需要学习的参数（在这个例子中是线性层的权重和偏置）**，**`lr=0.01` 是设置的学习率。**

优化器的作用是根据计算出的梯度来更新模型的参数。在训练过程中，每次迭代会执行以下步骤：
1. **`optimizer.zero_grad()`：清空过去的梯度。**
2. **`loss.backward()`：通过反向传播计算当前梯度。**
3. **`optimizer.step()`：根据梯度更新网络参数。**

#### 步骤 4: 训练模型

现在我们可以开始训练我们的模型了。

```python
# 训练模型
epochs = 1000  # 训练轮数
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清空过往梯度

    y_pred = model(x_data)  # 进行预测
	#print(y_pred == x_data @ model.linear.weight.t() +model.linear.bias )
	'''上述预测的本质是y=xw+b,上面的代码可以验证'''
    loss = loss_function(y_pred, y_data.unsqueeze(1))  # 计算损失

    loss.backward()  # 反向传播，计算当前梯度
    optimizer.step()  # 根据梯度更新网络参数

    # 每隔一段时间输出训练信息
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


#如果不好理解，可以和之前的对比
'''
# 执行手动梯度计算和更新
for _ in range(iterations):
    # 计算预测值
    y_pred = x_data @ weights + bias

    # 计算损失
    loss = mse_loss(y_pred, y_data)

    # 手动计算梯度
    grad_w = (2.0 / x_data.shape[0]) * (x_data.t() @ (y_pred - y_data))
    grad_b = (2.0 / x_data.shape[0]) * torch.sum(y_pred - y_data)

    # 更新权重和偏置
    weights -= learning_rate * grad_w
    bias -= learning_rate * grad_b
'''
```

这两段代码分别代表了使用 PyTorch 库和手动实现线性回归的训练过程。核心代码的对应关系可以这样解释：

1. **计算预测值**：
   - PyTorch: `y_pred = model(x_data)`
   - 手动: `y_pred = x_data @ weights + bias`

   这两行代码都是在计算模型的预测输出。PyTorch 版本中，`model(x_data)` 调用了模型的 `forward` 方法来计算预测值，而手动版本直接使用矩阵乘法和向量加法来进行计算。

2. **计算损失**：
   - PyTorch: `loss = loss_function(y_pred, y_data.unsqueeze(1))`
   - 手动: `loss = mse_loss(y_pred, y_data)`

   这里都是在计算预测值和真实值之间的损失，使用的是均方误差（Mean Squared Error, MSE）作为损失函数。在 PyTorch 版本中，`loss_function` 是预先定义的 MSE 损失函数，而在手动版本中，损失是通过 `mse_loss` 函数直接计算的。

3. **计算梯度**：
   - PyTorch: `loss.backward()`
   - 手动: `grad_w = (2.0 / x_data.shape[0]) * (x_data.t() @ (y_pred - y_data))` 和 `grad_b = (2.0 / x_data.shape[0]) * torch.sum(y_pred - y_data)`

   在 PyTorch 版本中，`loss.backward()` 自动计算损失相对于模型参数的梯度。在手动版本中，梯度是通过应用梯度公式直接计算的。

4. **更新模型参数**：
   - PyTorch: `optimizer.step()`
   - 手动: `weights -= learning_rate * grad_w` 和 `bias -= learning_rate * grad_b`

   在 PyTorch 版本中，`optimizer.step()` 根据计算出的梯度自动更新模型的权重和偏置。在手动版本中，权重和偏置的更新是通过直接应用梯度下降公式手动进行的。

总结来说，这两段代码在功能上是等价的：它们都实现了线性回归模型的训练过程，包括预测值的计算、损失的计算、梯度的计算和模型参数的更新。区别在于 PyTorch 版本利用了库中的功能来简化这些步骤，而手动版本则显式地实现了所有这些步骤。
**其他:**

1. `model.train()`: 这一行将模型设置为训练模式。这对于某些类型的层，如 `Dropout` 和 `BatchNorm` 等，是非常重要的，因为这些层在训练和评估（预测）模式下的行为是不同的。如果省略了这一行，而模型中又使用了这些类型的层，那么它们将不会以适当的方式（例如，启用 dropout 或使用批量统计数据而非移动平均）运行，这**可能会影响训练的效果**。如果模型很简单，例如只有线性层，那么 `model.train()` 可能不会有太大影响。

2. `optimizer.zero_grad()`: 在每次迭代中清空过往梯度是必要的。在PyTorch中，梯度是累加的，这意味着每次调用 `.backward()` 方法时，新计算的梯度会添加到已经存在的梯度上。**如果不清空梯度，那么每一轮的梯度就会与前一轮的梯度累加在一起**，导致梯度更新不正确。这可能会导致模型训练非常不稳定，甚至完全无法收敛。

#### 步骤 5: 评估模型

训练完成后，我们可以查看模型的参数，看看它们是否接近真实的权重和偏置。

```python
print("模型参数:", model.linear.weight.data, model.linear.bias.data)
```

#### 完整代码

以上所有代码片段合并起来，就构成了一个完整的多变量线性回归模型训练过程。

#### 步骤6：画图

对于两个特征变量的线性回归，您可以在三维空间中绘制一个平面来表示预测的模型，以及散点图来表示数据点。这可以使用 `matplotlib` 库中的 `mplot3d` 模块来完成。以下是示例代码：

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设模型已经训练完毕，并且我们有 model.linear.weight 和 model.linear.bias

# 创建一个新的图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
'''创建了一个新的 matplotlib 图形和一个 3D 子图。111 表示图形布局是 1x1 网格的第一个子图，projection='3d' 指定了子图是 3D 的。'''

# 绘制原始数据点
ax.scatter(x_data[:, 0].numpy(), x_data[:, 1].numpy(), y_data.numpy())
'''在 3D 空间中绘制原始数据点。x_data[:, 0] 和 x_data[:, 1] 是输入数据的两个特征，y_data 是目标值。(x_data[i,0],x_data[i,1],y_data[i])可以构成一个数据点'''

# 为了绘制平面，我们需要创建一个网格并计算相应的y值
x1_grid, x2_grid = torch.meshgrid(torch.linspace(-3, 3, 10), 
torch.linspace(-3, 3, 10))
'''torch.linspace(-3, 3, 10) 创建了从 -3 到 3 的均匀分布的 10 个点，torch.meshgrid 生成了对应的网格点。'''
y_grid = model.linear.weight[0, 0].item() * x1_grid + model.linear.weight[0, 1].item() * x2_grid + model.linear.bias.item()
'''计算了在每个网格点上的预测值。它使用了模型训练后的权重和偏置来计算每个点的预测 y 值。'''

# 绘制预测平面
ax.plot_surface(x1_grid.numpy(), x2_grid.numpy(), y_grid.numpy(), alpha=0.5)
'''绘制了预测的平面。plot_surface 用于绘制 3D 表面图，alpha=0.5 设置了表面的透明度。'''

# 设置坐标轴标签
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

# 显示图形
plt.show()
```

在这段代码中，我们首先绘制了数据点，然后创建了一个网格来代表特征空间，并使用模型的权重和偏置来计算网格上每个点的预测值，从而绘制了预测平面。

![image-20240211180135422](.\assets\image-20240211180135422.png)

**为什么是一个平面?**

在给定的代码中，`LinearRegressionModel` 类定义了一个简单的线性回归模型，它使用 `nn.Linear` 来创建一个线性层。这个线性层本质上是执行了以下数学运算：

$$ \hat{y} = Xw + b $$

其中：
- $ \hat{y} $ 是预测值。
- $ X $ 是输入特征。
- $ w $ 是模型的权重。
- $ b $ 是模型的偏置。

对于一个拥有两个特征的输入 $ X = [x_1, x_2] $，线性模型变为：

$$ \hat{y} = w_1x_1 + w_2x_2 + b $$

**这是一个平面方程在三维空间中的表示形式**。在这个方程中，$ w_1 $ 和 $ w_2 $ 是平面的方向系数，$ b $ 是截距项。因此，无论输入数据是什么，线性回归模型的输出都会形成一个平面。

**就和中学时的线性回归`y=kx+b`是拟合一条直线一样。**


#### 经典案例:波士顿房价预测

当然可以。让我们使用 PyTorch 来实现一个基于多元线性回归的简单神经网络。我们将使用一个常见的公开数据集，比如波士顿房价数据集（Boston Housing Dataset），这个数据集包含了波士顿地区的房屋价格及其相关的统计数据。

多元线性回归模型的目标是学习输入特征（如犯罪率、房产税率等）与目标值（房价）之间的线性关系。

首先，我们需要安装并导入必要的库：

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

然后，加载和预处理数据：

```python
# 加载波士顿房价数据
boston = load_boston()
X, y = boston.data, boston.target

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#指定测试集占数据集的比例为0.2,设置一个随机种子以确保结果的可重复性

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为 torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
```

接着，定义一个简单的多元线性回归模型：

```python
class LinearRegressionModel(nn.Module):
    def __init__(self, n_features):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(X_train.shape[1])
```

然后，设置损失函数和优化器：

```python
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
```

训练模型：

```python
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train)

    # 后向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
```

最后，评估模型性能：

```python
model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze()
    test_loss = criterion(predictions, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')
```

可能会有警告:
`load_boston函数已被弃用，将在未来的 scikit-learn 版本中被移除。原因是这个数据集涉及一些伦理问题，尤其是与社会经济地位和种族歧视相关的变量。`

#### 数据集

波士顿房价数据集（Boston Housing Dataset）是一个著名的数据集，常用于回归分析和机器学习的入门任务。这个数据集由美国人口普查服务局收集的数据构成，包含了波士顿地区的房价中位数和与之相关的统计数据。
用下面的代码可以查看到数据集描述:

```python
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.DESCR)
```

数据集中的每个样本包含以下特征：

1. **CRIM**: 城镇人均犯罪率。
2. **ZN**: 住宅用地超过 25,000 平方英尺的比例。
3. **INDUS**: 城镇非零售商业用地的比例。
4. **CHAS**: 查尔斯河虚拟变量（如果是河流边界，则为1；否则为0）。
5. **NOX**: 一氧化氮浓度。
6. **RM**: 住宅平均房间数。
7. **AGE**: 1940 年以前建成的自用房屋比例。
8. **DIS**: 与五个波士顿就业中心的加权距离。
9. **RAD**: 辐射状公路的可达性指数。
10. **TAX**: 每 10,000 美元的全值财产税率。
11. **PTRATIO**: 城镇师生比例。
12. **B**: 1000(Bk - 0.63)^2，其中 Bk 是城镇中黑人居民的比例。
13. **LSTAT**: 低收入人群的比例。

目标变量（`y`）是房屋的中位数价格（单位：千美元）。

**数据形状分析**

- `X_train` 和 `X_test` 的形状分别为 `(404, 13)` 和 `(102, 13)`。这表示训练集有 404 个样本，测试集有 102 个样本，每个样本有 13 个特征。
- `y_train` 和 `y_test` 的形状分别为 `(404,)` 和 `(102,)`。这表示训练集和测试集的目标变量是一维数组，与样本数量相对应。

#### 数据预处理

**数据标准化**过程中，每个特征的数据会经历以下变换：

1. **减去平均值**: 从每个特征的值中减去该特征的平均值。[对形状为 `(404, 13)` 的 `X_train` 进行标准化时，每个特征列（13个特征，每列代表一个特征）的每个元素都会减去该列的平均值。]
2. **除以标准差**: 将上一步的结果除以该特征的标准差。

数学上，标准化可以表示为：

$$ z = \frac{(x - \mu)}{\sigma}$$

其中 $ x $ 是原始值，$ \mu $ 是平均值，$ \sigma $ 是标准差，$ z $ 是标准化后的值。

这样做的目的是为了确保所有特征在模型训练中具有相同的规模和重要性(有的特征天生数值大，有的数值天生数值小)。如果特征的规模差异很大，那么在梯度下降等算法中，规模较大的特征可能会对模型训练产生不成比例的影响。
将数据从 **NumPy 数组转换为 PyTorch 张量（tensors）**。这是因为 PyTorch 用张量来表示数据，这些张量支持 GPU 加速等高效的计算操作。

**精度**: `float32` 是一种平衡了计算效率和精度的数据类型。对于大多数深度学习任务，`float32` 提供的精度已经足够。

**设计神经网络**

不再赘述，和之前完全一样,继承自nn，输入13个特征，输出1个特征的网络。

**定义损失函数**

不再赘述,经典MSELoss均方误差损失函数

**训练**

不再赘述,迭代一千次，每次四个步骤:用训练数据进行预测，和实际值对比计算损失，计算损失对权重的梯度，梯度下降。其中记得zero_grad()保证梯度正确计算，每迭代100次输出预测误差。

**评估**

1. **设置模型为评估模式 (`model.eval()`)**:
    - 这一步通过调用 `eval()` 方法将模型设置为评估模式。
    - 在评估模式下，所有专门针对训练过程的操作（如 dropout 和批量归一化）会被禁用。
    - 这是因为在评估模型时，我们希望模型表现出其已学习的固定模式，而非继续学习或调整。

2. **禁用梯度计算 (`with torch.no_grad():`)**:
    - `torch.no_grad()` 语句用于暂时禁用梯度计算。
    - 在评估模型时，不需要计算梯度，这可以减少内存消耗并提高计算速度。
    - 这是因为评估阶段不涉及模型参数的更新。

3. **进行预测 (`predictions = model(X_test).squeeze()`)**:
   
    - 模型使用测试数据 (`X_test`) 进行预测。
    - `squeeze()` 方法去除可能存在的多余维度，使得 `predictions` 的维度与 `y_test` 一致。
    
      ```python
      print(predictions.size(),y_test.size(),predictions, y_test)
      '''torch.Size([102]) torch.Size([102])'''
      ```
    
4. **计算测试损失 (`test_loss = criterion(predictions, y_test)`)**:
    - 用均方误差计算损失。
    
    $$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
    
    MSE 作为损失函数的一个重要特性是，它会对较大的误差赋予更高的惩罚（由于平方项的影响），这有助于模型更加关注和减少大的预测误差。

### 分类任务

#### One-hot 编码

对于多分类问题，one-hot 编码是一种将类别标签转换为二进制（0和1）形式的方法。每个标签转换为一个与类别数量相等长度的向量，其中真实类别对应的元素设为 1，其余设为 0。

在我们的例子中，假设有三个类别，真实类别是第一个类别(比如说一张照片可以是猫、狗、人，实际上是猫)，所以 one-hot 编码的标签将是:

```plaintext
one-hot label: [1, 0, 0]
```

#### Softmax的概率意义

$softmax(xi) = \frac{e^{xi}}{\sum_{j} e^{xj}}$

```python
import torch

# 假设我们有一个三类分类问题的 logits【在机器学习和深度学习领域，"logits"这个术语通常指模型输出层之前的原始预测值，也就是未经过归一化（如 softmax 函数）处理的预测值。】
logits = torch.tensor([[2.0, 1.0, 0.1]])
print(torch.softmax(logits, dim=1))
'''tensor([[0.6590, 0.2424, 0.0986]])'''
```

解释 Softmax 输出

- **softmax(logits, dim=1)**：这个函数将 `logits` 张量中的值转换为概率。`dim=1` 指定了 softmax 函数沿着张量的第二个维度（即每行内部）进行计算。

- **输出的含义**：softmax 函数的输出是一个概率分布，它表示模型对每个类别的预测概率。在这个例子中，输出张量 `[[0.6590, 0.2424, 0.0986]]` 表示模型预测第一个类别的概率为约 65.9%，第二个类别的概率为约 24.24%，第三个类别的概率为约 9.86%。

概率解释

1. **第一个类别（65.9%）**：模型认为输入最有可能属于第一个类别。
2. **第二个类别（24.24%）**：第二个类别是模型的次要选择。
3. **第三个类别（9.86%）**：模型认为输入属于第三个类别的可能性最低。

在多类分类问题中，softmax 函数确保输出的概率总和为 1，并且每个类别的概率都介于 0 到 1 之间。这使得 softmax 函数成为分类问题中常用的激活函数。在这种情况下，通常选择概率最高的类别作为模型的最终预测。

#### Log Softmax 

```python
import torch

# 假设我们有一个三类分类问题的 logits
logits = torch.tensor([[2.0, 1.0, 0.1]])

# 计算 Softmax
softmax_probs = torch.exp(logits) / torch.sum(torch.exp(logits), dim=1, keepdim=True)

# 计算 Log Softmax
log_softmax_manual = torch.log(softmax_probs)

print("Log Softmax (manual, unstable):", log_softmax_manual)
print("Log Softmax (PyTorch):", torch.log_softmax(logits, dim=1))
'''Log Softmax (PyTorch): tensor([[-0.4170, -1.4170, -2.3170]])'''

```

![image-20240220125644611](.\assets\image-20240220125644611.png)

#### 计算交叉熵损失

对于给定的例子，我们有3个类别的logits，并且已经计算出了Log Softmax的值。现在，假设实际的类别标签是0，1，和2，我们来计算每种情况下的交叉熵损失。

**交叉熵损失的计算公式为：**

$$ L = -\sum_{c=1}^{M} y_{o,c} \log(p_{o,c}) $$

其中，

- $M$ 是类别的数量，在这个例子中是3。

- $y$ 是一个二元指示器（indicator）数组，如果类别 $c$ 是正确的分类，则 $y_{o,c} = 1$，否则为0。

- $p$ 是预测的概率分布，由Softmax计算得到。

- $o$ 是数据点的索引，在这个例子中，我们只有一个数据点。

  ![image-20240320144620122](./assets/image-20240320144620122.png)

**计算实际的分类0，1，2的交叉熵损失**

假设实际分类是0：

$$ L = -\log(p_{0,0}) $$

根据PyTorch计算的Log Softmax值，对于类别0，Log Softmax值是$-0.4170$，所以交叉熵损失是：

$$ L = -(-0.4170) = 0.4170 $$

如果实际的分类是1，交叉熵损失为1.417，如果实际的分类是2，交叉熵损失为2.317。

#### 为什么这样设计？

交叉熵损失的设计考虑了概率分布的特性：**当预测概率$p_{o,c}$接近实际概率$y_{o,c}$时，损失越小；当预测概率与实际概率差距大时，损失越大。特别是，当实际标签的预测概率接近1时（即预测非常准确），损失接近0；当实际标签的预测概率很小（即预测不准确），损失会很大。**

![image-20240320144708189](./assets/image-20240320144708189.png)

**公式当中为什么有求和符号？**

**求和符号使得损失函数可以无缝地应用于多标签分类问题**。在多标签分类中，一个实例可以同时属于多个类别，这意味着独热编码向量中可能有多个1。因此，求和符号确保了所有正确类别的负对数概率都被计算在内。

**极端情况**
**交叉熵为0的情况发生在模型对每个实例的分类完全正确且确信无疑的情况下。**这意味着对于正类实例，logits $z$ 应无限大接近正无穷；对于负类实例，logits $z$ 应无限大接近负无穷。实际上，这种情况在实践中几乎不可能达到，因为它要求模型对所有实例的分类都绝对准确且完全确定。

#### 二分类

![image-20240320151834099](./assets/image-20240320151834099.png)

当交叉熵损失应用于二分类问题时，公式可以简化成一个非常熟悉的形式，这种形式特别是在逻辑回归中常见，也被称为二元交叉熵损失（Binary Cross-Entropy Loss）。这时，输出可以通过一个单一的概率值 $p$ 来表示，其中 **$p$ 表示样本属于类别 1 的预测概率**（相应地，$1-p$ 表示样本属于类别 0 的预测概率）。假设 **$y$ 是实际的标签**，其中 $y=1$ 表示正类，$y=0$ 表示负类，那么二元交叉熵损失的公式可以表示为：

$$ L = -[y \log(p) + (1 - y) \log(1 - p)] $$

这个公式的含义是：
- **当实际标签 $y=1$ 时，损失函数简化为 $L = -\log(p)$，这意味着如果模型对正类的预测概率 $p$ 非常确定（接近 1），那么损失将会很小；如果模型对正类的预测概率不确定（$p$ 接近 0），那么损失将会很大。**
- **相反地，当实际标签 $y=0$ 时，损失函数简化为 $L = -\log(1 - p)$。这时，如果模型对负类的预测概率 $1-p$ 非常确定（即 $p$ 接近 0），那么损失将会很小；如果模型对负类的预测概率不确定（即 $p$ 接近 1），那么损失将会很大。**

二元交叉熵损失这样设计的目的是为了在二分类问题中有效地衡量模型预测的准确性。它直接对模型的输出概率进行惩罚，强调了对正确类别的概率预测的准确性。这种损失函数非常适合处理输出概率的模型，如逻辑回归和具有sigmoid激活函数的输出层的神经网络。

注：在机器学习和深度学习的上下文中，当我们提到对数（log）函数，尤其是在计算Softmax函数的对数形式（即Log Softmax）时，我们通常指的是自然对数，其底数是 $e$（约等于2.71828）。因此，当我们说“log”时，实际上是指“ln”（自然对数）。在编程实现中，例如使用Python的NumPy或PyTorch库时，`log`函数（比如`numpy.log`或`torch.log`）也是指计算自然对数。

**总结**

- **二分类问题**：是多分类问题的一个特殊情况，类别总数为2。
- **多分类问题**：每个实例只能被分到一个类别中，是多标签分类问题的一个特殊情况，其中每个实例恰好被分到一个类别中。
- **多标签分类问题**：每个实例可以属于多个类别。

### 卷积神经网络

![image-20240211193727742](.\assets\image-20240211193727742.png)

#### MNIST 数据集

一个经典的卷积神经网络（CNN）例子是使用 PyTorch 在 MNIST 数据集上训练一个简单的网络。**MNIST 是一个包含手写数字（0到9）的大型数据库**，常用于训练各种图像处理系统。

下面是一个基本的 CNN 结构，用于识别 MNIST 数据集中的手写数字。我会提供代码示例和每一部分的简要说明：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 定义 CNN 模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 实例化模型、定义损失函数和优化器
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
	    #print(images.size(),labels.size())
	    '''torch.Size([64, 1, 28, 28]) torch.Size([64])'''
		# print(labels)
		'''
		tensor([4, 5, 1, 1, 7, 5, 3, 5, 7, 4, 6, 4, 7, 9, 2, 4, 5, 8, 1, 7, 3, 5, 1, 6,
        3, 1, 9, 3, 3, 4, 6, 2, 4, 0, 2, 5, 3, 1, 1, 8, 6, 1, 6, 6, 9, 0, 0, 2,
        6, 4, 8, 3, 8, 0, 5, 5, 6, 3, 4, 0, 1, 5, 1, 3])
		'''
        # 前向传播
        outputs = model(images)
        # print(outputs.size())
		'''torch.Size([64, 10])'''
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')
```

这个例子包括以下部分：
1. **模型定义**：定义了一个简单的 CNN，包含两个卷积层，两个全连接层。
2. **数据加载**：使用 torchvision 加载并预处理 MNIST 数据集。
3. **训练循环**：进行前向传播、计算损失、反向传播以及优化步骤。
4. **测试模型**：在测试集上评估模型的准确度。

请确保您已经安装了 PyTorch 和 torchvision(之前的anaconda已经安装了)。您可以直接运行这个脚本来训练模型。这只是一个起点，您可以根据需要调整模型的结构、超参数等。

#### 加载数据集

MNIST 数据集存储在 `.gz` 文件中，这是 gzip 压缩格式。当解压后，数据以 IDX 文件格式存储，这是一个用于向量和多维矩阵的文件格式，通常用于存储大量的数据。

MNIST 数据集的图像是**黑白图像（单通道），每张图像的大小是 28x28 像素**。在 PyTorch 中，图像会被转换为 **`FloatTensor`，其形状为 `[batch_size, channels, height, width]`**。对于 MNIST 数据集，这将是 **`[batch_size, 1, 28, 28]`**。这里的 `batch_size` 取决于您在 `DataLoader` 中定义的大小，`channels` 是 1（因为是黑白图像），`height` 和 `width` 都是 28。

![image-20240211193543384](.\assets\image-20240211193543384.png)

1. **定义数据转换**：
   
   ```python
   transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
   ```
   `transforms.Compose` 是一个组合类，它将多个变换组合在一起。在这个特定的例子中，包含了两个变换：`transforms.ToTensor()` 和 `transforms.Normalize((0.5,), (0.5,))`。
   
   1. **`transforms.ToTensor()`**：
      
      - **原始数据**：MNIST 数据集中的原始图像是 PIL 图像格式，每个像素的值是一个从 0 到 255 的整数。
      - **类型转换**：`transforms.ToTensor()` 会将 PIL 图像或 NumPy `ndarray` 转换为 PyTorch 的 `FloatTensor`。
      - **数值转换**：
        $$
        \text{normalized\_value} = \frac{\text{original\_value}}{255}
        $$
        即将原始像素值除以 255，使得像素值范围从 [0, 255] 缩放到 [0.0, 1.0]。
      - **结果**：转换后的张量的形状是 `(C, H, W)`，其中 `C` 是通道数（对于 MNIST 是 1，因为它是灰度图像），`H` 是图像的高度，`W` 是图像的宽度。同时，像素值会被缩放到 `[0, 1]` 的范围内，即原来的整数 `0-255` 被转换成了浮点数 `0.0-1.0`。
      
   2. **`transforms.Normalize((0.5,), (0.5,))`**：
      
      - **原始数据**：此时，输入数据是经过 `ToTensor()` 转换后的，像素值范围为 `[0, 1]` 的张量。
      - **归一化公式**：归一化操作是按通道执行的，公式为：
        $$ \text{output[channel]} = \frac{\text{input[channel]} - \text{mean[channel]}}{\text{std[channel]}} $$
      - **转换过程**：归一化操作会对输入数据的每个通道执行以下操作：
        其中 `mean` 和 `std` 是预先定义的均值和标准差。在这个例子中，由于是单通道图像，我们只有一个均值和一个标准差。在许多情况下，如果数据近似在 [0, 1] 的范围内，则均值和标准差常常会选择为 0.5。
      - **结果**：因此，**每个像素值会先减去 0.5，然后除以 0.5。这意味着经过归一化后的数据将有一个大约范围为 `[-1, 1]`** 的新的均值和标准差。
   
   综合来看，这个转换管道将图像数据从 `[0, 255]` 的整数值缩放并转换成了 `[0, 1]` 的浮点数值，并进一步将其标准化到大约 `[-1, 1]` 的范围。这样的预处理步骤通常是因为**归一化后的数据对于模型的训练来说收敛更快，性能也更好**。
   
2. **加载训练和测试数据集**：
   
   ```python
   train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
   test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
   ```
   这两行代码分别**加载训练集和测试集。**
   - `datasets.MNIST` 是一个封装好的数据集类，它能自动处理 MNIST 数据集的下载和加载。
   - **`root='./data'` 指定了数据集的保存位置，如果该位置没有数据集，则会自动下载。**
   - `train=True` 或 `train=False` 指定了是加载训练集还是测试集。
   - `download=True` 告诉程序如果数据集不在 `root` 指定的路径下，则需要从互联网上下载数据集。
   - `transform=transform` 应用之前定义的转换到数据集的每个元素。
   
3. **创建数据加载器**：
   ```python
   train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
   test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
   ```
   这两行代码创建了数据加载器，用于迭代地加载数据集。
   - `DataLoader` 是 PyTorch 中的一个类，它提供了对 `Dataset` 的封装，方便批量加载数据，并且可以提供多进程加速。
   - `dataset=train_dataset` 或 `dataset=test_dataset` 指定了要加载的数据集。
   - **`batch_size=64` 指定了每个批次加载多少样本。**
   - `shuffle=True` 或 `shuffle=False` 指定了是否在每个 epoch 开始时打乱数据。

至于数据集的来源，`torchvision` 的 `datasets` 类已经预定义了常用数据集的下载 URL。因此，当你调用 `datasets.MNIST` 并设置 `download=True` 时，它会自动从预定义的 URL 下载数据集。

文件名也是由 `datasets.MNIST` 类内部处理的。它知道 MNIST 数据集的文件结构和文件名，所以用户不需要指定这些细节。这使得下载和加载数据变得非常简单和方便。

#### 设计神经网络

```python
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
	    #print(images.size(),labels.size())
	    '''torch.Size([64, 1, 28, 28]) torch.Size([64])'''
		# print(labels)
		'''
		tensor([4, 5, 1, 1, 7, 5, 3, 5, 7, 4, 6, 4, 7, 9, 2, 4, 5, 8, 1, 7, 3, 5, 1, 6,
        3, 1, 9, 3, 3, 4, 6, 2, 4, 0, 2, 5, 3, 1, 1, 8, 6, 1, 6, 6, 9, 0, 0, 2,
        6, 4, 8, 3, 8, 0, 5, 5, 6, 3, 4, 0, 1, 5, 1, 3])
		'''
        # 前向传播
        outputs = model(images)
        # print(outputs.size())
		'''torch.Size([64, 10])'''
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

![image-20240212155642744](.\assets\image-20240212155642744.png)

1. **初始输入图像**：
   - 形状为 `[batch_size, 1, 28, 28]`。`batch_size` 是批次中的图像数量，这里设为 64。1 是图像的通道数（灰度图），28x28 是图像的高度和宽度。

2. **第一个卷积层 (`conv1`)**：
   - `nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)` 应用了 32 个过滤器（卷积核），每个核的大小为 3x3，步长为 1，边缘填充为 1。
   - 输出形状变为 `[batch_size, 32, 28, 28]`。这是因为使用了 padding，所以尽管应用了卷积，图像大小保持不变，但是通道数增加到了 32。

3. **第一个池化层 (`pool`)**：
   - `nn.MaxPool2d(kernel_size=2, stride=2)` 应用了一个大小为 2x2，步长为 2 的最大池化操作。
   - 输出形状变为 `[batch_size, 32, 14, 14]`。池化操作减小了图像的高度和宽度的尺寸，每个维度都减半。

![image-20240212155848067](.\assets\image-20240212155848067.png)

1. **第二个卷积层 (`conv2`)**：
   - `nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)` 应用了 64 个过滤器，每个核的大小为 3x3，步长为 1，边缘填充为 1。
   - 输出形状变为 `[batch_size, 64, 14, 14]`。通道数增加到了 64，图像大小由于 padding 保持不变。

2. **第二个池化层 (`pool`)**：
   - 使用相同的最大池化操作。
   - 输出形状变为 `[batch_size, 64, 7, 7]`。再次减小图像的高度和宽度。

![image-20240212155936501](.\assets\image-20240212155936501.png)

1. **展平操作**：
   - `torch.flatten(x, 1)` 将每个批次中的图像从三维（通道、高度、宽度）展平为一维，以便能够作为全连接层（线性层）的输入。
   - 输出形状变为 `[batch_size, 64*7*7]`，或者说 `[batch_size, 3136]`，因为 64x7x7 = 3136。

2. **第一个全连接层 (`fc1`)**：
   - `nn.Linear(64 * 7 * 7, 1000)` 将展平后的张量连接到 1000 个神经元上。
   - 输出形状变为 `[batch_size, 1000]`。

3. **第二个全连接层 (`fc2`)**：
   - `nn.Linear(1000, 10)` 进一步将特征从 1000 个维度减少到 10 个维度，对应于 10 个类别的数字（0-9）。
   - 输出形状变为 `[batch_size, 10]`。

在这个网络中，卷积层负责提取图像的特征，池化层负责减少特征的空间尺寸（降维），而全连接层负责将这些特征映射到最终的分类结果。

#### outputs 与 labels 形状的匹配

在代码中，`outputs` 的形状是 `[64, 10]`，表示每个批次有 64 个样本，每个样本有 10 个类别的预测分数。`labels` 的形状是 `[64]`，表示每个样本的真实类别索引。

`CrossEntropyLoss` 不需要 `labels` 是 one-hot 编码的形式，它只需要类别的索引。因此，尽管 `outputs` 和 `labels` 的第二维大小不同，但 `CrossEntropyLoss` 会自动处理，只考虑 `outputs` 中对应 `labels` 索引的 logits 来计算损失。

#### 训练神经网络的通用步骤

训练神经网络的四个基本步骤如下：

1. **前向传播**：通过模型传递数据以获得预测输出。
2. **计算损失**：使用损失函数比较预测输出和真实标签，计算损失值。
3. **反向传播**：通过损失函数反向传递损失，计算每个参数的梯度。
4. **优化步骤**：使用优化器（如 SGD）根据计算的梯度更新模型的参数。`optimizer.step()` 实际上执行了这一步。虽然梯度下降是最常见的优化方法之一，但 `optimizer.step()` 可以执行更复杂的优化算法，如 SGD 的变体（带动量的 SGD）、Adam 或 RMSprop 等，这些算法可能包括梯度的平滑、自适应学习率等。

#### 打乱顺序的多轮训练

在 PyTorch 的 `DataLoader` 中设置 `shuffle=True` 意味着在每个训练时代（epoch）开始时，训练数据集的样本顺序会被打乱。这样做有几个好处：

1. **防止过拟合**：如果每次迭代都使用相同的样本顺序，模型可能会对特定的样本顺序产生依赖，这可能会导致模型学习到数据中的噪声。打乱顺序有助于防止模型对训练数据的特定顺序过度拟合。

2. **提高泛化能力**：随机化样本顺序有助于模型学习更加普遍的特征，这通常可以提高模型在未见过的数据上的表现，即提高泛化能力。

3. **优化收敛**：在一些情况下，如果训练样本有特定的顺序（例如，按类别排序），这可能会导致梯度下降过程陷入局部最小点。打乱顺序可以帮助优化算法更好地探索参数空间。

#### 测试集评估

**将模型设置为评估模式**：

1. ```python
   model.eval()
   ```
   这一步通过调用 `eval()` 函数将模型设置为评估模式。在评估模式下，某些特定于训练阶段的层（如 Dropout 和 BatchNorm）将调整其行为。例如，Dropout 将不再丢弃任何单元，BatchNorm 将使用在训练过程中学习的运行平均值和方差。
   
2. **关闭梯度计算**：
   ```python
   with torch.no_grad():
   ```
   使用 `torch.no_grad()` 上下文管理器禁用梯度计算。在测试过程中，我们不需要计算梯度，这可以减少内存使用并加速计算。

3. **遍历测试数据集**：
   ```python
   for images, labels in test_loader:
   ```
   使用 `test_loader` 遍历测试数据集。`test_loader` 会按批次提供图像和对应的真实标签。

![image-20240212165107294](.\assets\image-20240212165107294.png)

1. **前向传播和预测**：
   ```python
   outputs = model(images)
   _, predicted = torch.max(outputs.data, 1)
   ```
   对每个批次的图像进行**前向传播，得到模型的预测输出`outputs`**。
   `torch.max` 返回两个值：

   - 第一个值是每个样本中找到的最大概率值（在这个上下文中，我们通常不关心这个值）。
   - 第二个值predicted是这些最大概率值所对应的索引，即预测的类别标签。在十类分类问题中，这些索引将是介于 0 到 9 之间的整数。

   函数**沿维度 1（类别的维度）寻找最大值**。这意味着对于每个样本（64个中的每一个），它都会在这10个类别预测概率中找到最大的一个。
   **`predicted`是一个[64]的向量。**

2. **统计正确预测的数量**：

   ```python
   total += labels.size(0)
   correct += (predicted == labels).sum().item()
   ```
   `total` 变量记录了总的样本数，而 `correct` 变量记录了正确预测的样本数。`predicted == labels` 创建了一个布尔数组，表示每个样本是否被正确分类，然后通过 `.sum().item()` 计算了正确分类的总数。

3. **计算准确率**：
   ```python
   print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')
   ```
   最后，计算并打印模型在整个测试集上的准确率。准确率是正确预测的数量除以总样本数。

#### 准确率≈99%

- 训练数据通过 `DataLoader` 加载时使用了 `shuffle=True`，这意味着在每个训练周期中，训练数据的顺序都会被打乱。这有助于提高模型的泛化能力，但也会导致每次训练的过程略有不同。
- 另外，模型（随机）初始化和某些层也可能引入随机性。

- 使用随机梯度下降（SGD）作为优化器，它在优化过程中会利用随机选取的小批量数据计算梯度，这也可能导致训练过程中的随机性。

#### 使用GPU训练

![image-20240215131645659](.\assets\image-20240215131645659.png)

刚才的代码cpu跑满，但是gpu没有使用。

```python
# 引入必要的库

# 检查是否有可用的 CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
'''cuda:0'''

# 定义 CNN 模型
class ConvNet(nn.Module):
    # ...

# 数据预处理、加载 CIFAR-10 数据集
# ...

# 实例化模型、定义损失函数和优化器
model = ConvNet().to(device)  # 将模型移到 GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)  # 将数据移到 GPU
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ...

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # 将数据移到 GPU
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # ...

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')

```

占用GPU，速度提升了非常多倍
![image-20240215131918457](.\assets\image-20240215131918457.png)

#### 卷积神经网络-图像分类

一个使用 CIFAR-10 数据集的卷积神经网络（CNN）例子。CIFAR-10 数据集包含 60000 张 32x32 像素的**彩色图像，分为 10 个类别**，每个类别有 6000 张图像。数据集被分为 50000 张训练图像和 10000 张测试图像。

以下是使用 PyTorch 实现的一个简单的 CNN 来处理 CIFAR-10 数据集的例子。这个网络会稍微复杂一些，以适应 CIFAR-10 数据集的复杂性。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义 CNN 模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(64 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out, 1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载 CIFAR-10 数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 实例化模型、定义损失函数和优化器
model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')
```

这个例子中，网络架构包含两个卷积层，每个卷积层后面跟着批量归一化层（BatchNorm2d）、ReLU激活函数和最大池化层（MaxPool2d）。接着是两个全连接层。这个网络比之前的 MNIST 网络更复杂，以适应 CIFAR-10 数据集的更高复杂度。训练和测试的步骤与之前的例子相似，但需要注意的是，CIFAR-10 的图像是彩色的，所以第一个卷积层的输入通道数是 3（代表 RGB 三个颜色通道）。

#### 加载数据集

![image-20240213132249452](.\assets\image-20240213132249452.png)

![image-20240213132407826](.\assets\image-20240213132407826.png)

1. **数据转换（transforms）**：
   - 对于 **CIFAR-10** 数据集，使用的是三个通道的归一化，因为 CIFAR-10 是彩色图像（包含 RGB 三个通道）。`transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` 用于将每个通道的像素值标准化，使其大致分布在 [-1, 1] 的范围内。
   - 对于 **MNIST** 数据集，由于它是灰度图像（只有一个通道），因此使用的是单通道的归一化。

其他代码相同，不再介绍。

#### 网络结构和张量分析

这个CNN 架构中，网络包括两个卷积层、两个池化层和两个全连接层。让我们逐层分析其结构及输出的张量形状：
![image-20240214204534601](.\assets\image-20240214204534601.png)

1. **输入图像**：
   - 形状：`[64, 3, 32, 32]`
   - 解释：批量大小为 64，每个图像有 3 个颜色通道（RGB），图像尺寸为 32x32 像素。

2. **第一层（layer1）**：
   - 结构：Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
   - `Conv2d(3, 32, kernel_size=3, padding=1)`：从 3 通道到 32 通道的卷积，核大小为 3x3，边缘填充为 1。这保持了图像的空间尺寸不变（32x32）。
   - `BatchNorm2d(32)`：批量归一化，作用于 32 个输出通道。
   - `ReLU`：激活函数。
   - `MaxPool2d(kernel_size=2, stride=2)`：2x2 的最大池化，减半图像尺寸。
   - 输出形状：`[64, 32, 16, 16]`（批量大小，通道数，高度，宽度）

![image-20240214204657429](.\assets\image-20240214204657429.png)

1. **第二层（layer2）**：
   - 结构：Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
   - `Conv2d(32, 64, kernel_size=3, padding=1)`：从 32 通道到 64 通道的卷积，核大小为 3x3，边缘填充为 1。这保持了图像的空间尺寸不变（16x16）。
   - `BatchNorm2d(64)`：批量归一化，作用于 64 个输出通道。
   - `ReLU`：激活函数。
   - `MaxPool2d(kernel_size=2, stride=2)`：2x2 的最大池化，减半图像尺寸。
   - 输出形状：`[64, 64, 8, 8]`

![image-20240214205000261](.\assets\image-20240214205000261.png)

1. **展平操作**：
   - 将每个批次中的图像从三维（通道、高度、宽度）展平为一维。
   - 输出形状：`[64, 64*8*8]` 或 `[64, 4096]`

2. **第一个全连接层（fc1）**：
   - `Linear(64 * 8 * 8, 1000)`：将 4096 维的输入连接到 1000 个神经元上。
   - 输出形状：`[64, 1000]`

3. **第二个全连接层（fc2）**：
   - `Linear(1000, 10)`：将 1000 维的输入连接到 10 个输出神经元上，对应于 CIFAR-10 的 10 个类别。
   - 输出形状：`[64, 10]`

**训练和测试和之前手写数字识别一样，不再赘述**

### 1维卷积



![image-20240322070912312](./assets/image-20240322070912312.png)



![image-20240322070950382](./assets/image-20240322070950382.png)

```py
import torch
import torch.nn.functional as F


# 创建一个Conv1d层并提取其权重和偏置
conv1d_layer = torch.nn.Conv1d(in_channels=300, out_channels=64, kernel_size=3, stride=1, padding=1)
conv1d_weight = conv1d_layer.weight.data
conv1d_bias = conv1d_layer.bias.data


# 创建一个输入张量
input_tensor = torch.randn(2, 300, 100)  # 假设有1个样本，每个样本有300个通道，宽度为100

# 使用PyTorch的Conv1d
output_conv1d = conv1d_layer(input_tensor)

#获取权重和偏置
weights = conv1d_layer.weight.data
bias = conv1d_layer.bias.data

# 手动实现卷积的函数
def manual_conv1d(input_tensor, weights, bias, stride=1, padding=1):
    batch_size, in_channels, width = input_tensor.shape
    out_channels, _, kernel_size = weights.shape

    # 计算输出宽度
    output_width = ((width + 2 * padding - kernel_size) // stride) + 1

    # 应用padding
    if padding > 0:
        input_padded = F.pad(input_tensor, (padding, padding), "constant", 0)
    else:
        input_padded = input_tensor

    # 初始化输出张量
    output = torch.zeros(batch_size, out_channels, output_width)

    # 执行卷积操作
    for i in range(out_channels):
        for j in range(output_width):
            start = j * stride
            end = start + kernel_size
            # 对所有输入通道执行卷积并求和
            output[:, i, j] = torch.sum(input_padded[:, :, start:end] * weights[i, :, :].unsqueeze(0), dim=(1, 2)) + \
                              bias[i]

    return output

print("output_conv1d:", output_conv1d)

# 应用手动卷积
manual_conv1d_output = manual_conv1d(input_tensor, weights, bias, stride=1, padding=1)
print("manual_conv1d_output:", manual_conv1d_output)

# 比较结果
print("Output close:", torch.allclose(output_conv1d, manual_conv1d_output, atol=1e-4))

```



![image-20240322071010895](./assets/image-20240322071010895.png)





### NLP初步

#### 中文分词工具

`jieba` 是一个非常流行的中文分词库，广泛用于中文自然语言处理。它支持三种分词模式：精确模式、全模式和搜索引擎模式，并且可以处理简体和繁体中文。以下是关于 `jieba` 的一些基本信息和使用方法：

#### 安装

在使用 `jieba` 之前，需要先进行安装，可以通过 pip 安装：
```bash
pip install jieba
```

**指令在哪里输入**

可以直接在pycharm的终端输入，如果打不开终端可以搜索Anaconda Prompt,之后输入即可。
![image-20240215185210227](.\assets\image-20240215185210227.png)

![image-20240215185242600](.\assets\image-20240215185242600.png)

#### 基本用法

1. **精确模式**：这种模式下，`jieba` 尝试将句子最精确地切开，适合文本分析。

   ```python
   import jieba

   seg_list = jieba.cut("jieba是一个非常流行的中文分词库广泛用于中文自然语言处理", cut_all=False)
   print("精确模式: " + "/ ".join(seg_list))
   ```

2. **全模式**：在这种模式下，`jieba` 会将句子中所有可能的词语都扫描出来，速度非常快，但不适合文本分析。

   ```python
   seg_list = jieba.cut("jieba是一个非常流行的中文分词库广泛用于中文自然语言处理", cut_all=True)
   print("全模式: " + "/ ".join(seg_list))
   ```

3. **搜索引擎模式**：在这种模式下，`jieba` 会对长词再次切分，提高召回率，适用于搜索引擎构建索引。

   ```python
   seg_list = jieba.cut_for_search("jieba是一个非常流行的中文分词库广泛用于中文自然语言处理")
   print("搜索引擎模式: " + "/ ".join(seg_list))
   ```

#### 繁体分词

`jieba` 同样支持繁体中文的分词。它的分词算法和词典是针对简体中文优化的，但对于繁体中文，表现也相当不错。使用方法与简体中文相同：

```python
seg_list = jieba.cut("jieba是壹個非常流行的中文分詞庫廣泛用于中文自然語言處理", cut_all=False)
print("精确模式: " + "/ ".join(seg_list))
```

#### 自定义词典

`jieba` 允许用户添加自定义词典，以适应特定领域的分词需要。这对于一些特殊名词或者新词尤其有用。

```python
jieba.load_userdict("./data/Tokenization/userdict.txt")
```

其中 `userdict.txt` 是一个自定义词典文件，其内容格式如下：

```
词语 词频 词性(可选)
```

例如：

```python
分词库 1000 n
```

#### HanLP

**安装 HanLP**

HanLP 是一款由自然语言处理专家开发的多语言处理库，支持多种语言，包括中文和英文。要在 Python 中安装 HanLP，您可以按照以下步骤操作：

1. **安装库**:
   使用 pip 命令安装 HanLP。在命令行（例如 Anaconda Prompt 或系统的命令行界面）中输入以下命令：

   ```bash
   pip install hanlp
   ```

![image-20240219111459365](.\assets\image-20240219111459365.png)

1. **检查环境**:
   确保您的 Python 环境中安装了 Java 运行时环境（JRE），因为 HanLP 的某些功能依赖于 Java。您可以在命令行中运行 `java -version` 来检查是否安装了 Java。
   ![image-20240219111529561](.\assets\image-20240219111529561.png)

使用 HanLP 进行中英文分词

安装完成后，您可以使用 HanLP 进行中英文分词。以下是一个简单的示例：

```python
import hanlp

# 初始化分词器
tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

# 中文分词
text_cn = "今天天气真好，我们一起去公园散步吧。"
tokens_cn = tokenizer(text_cn)
print("中文分词:", tokens_cn)

# 英文分词
text_en = "Today is a good day, let's go to the park for a walk."
tokens_en = tokenizer(text_en)
print("英文分词:", tokens_en)

'''
中文分词: ['今天', '天气', '真', '好', '，', '我们', '一起', '去', '公园', '散步', '吧', '。']
英文分词: ['Today', 'is', 'a', 'good', 'day', ',', "let's", 'go', 'to', 'the', 'park', 'for', 'a', 'walk', '.']
'''
```

附录：
如果下载出错，可以尝试手动下载模型文件到指定的路径。访问提供的 URL类似于 (`http://download.hanlp.com/tok/coarse_electra_small_20220616_012050.zip`)，手动下载 zip 文件。将下载的文件解压到指定的目录（例如 `C:\Users\86157\AppData\Roaming\hanlp\tok`）。

#### 命名实体识别[略]

使用 HanLP 进行中文命名实体识别（NER，Named Entity Recognition）也是相对直接的。**命名实体识别是指识别文本中具有特定意义的实体，如人名、地名、机构名等**。以下是使用 HanLP 进行中文命名实体识别的基本步骤：

步骤 1：安装并导入 HanLP

如果您还没有安装 HanLP，可以通过 pip 进行安装，这里可能环境依赖难以解决，可以用：
```bash
pip install hanlp[full] -U
```

然后在您的 Python 脚本中导入 HanLP：
```python
import hanlp
```

步骤 2：加载预训练的命名实体识别模型

HanLP 提供了多种预训练模型，包括用于命名实体识别的模型。您可以加载适用于您需求的模型。例如，加载一个通用的中文 NER 模型：
```python
# 这里选择一个适用的预训练模型
recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
```

步骤 3：使用模型进行命名实体识别

使用加载的模型对中文文本进行命名实体识别：
```python
text = "汉克斯出生于加州的康科德市，他的父亲是厨师，母亲是医院工作者。"
entities = recognizer(text)
print(entities)
```

这将输出文本中识别的命名实体及其类别。

结果类似于:
```python
[('汉克斯', '人名'), ('加州', '地名'), ('康科德市', '地名')]
```

#### 词性标注

`jieba` 是一个广泛使用的中文分词工具，它也支持词性标注功能。词性标注是指为文本中的每个词分配一个词性（如名词、动词等）。以下是如何使用 `jieba` 进行中文词性标注的步骤：

安装 `jieba`

如果您还没有安装 `jieba`，可以通过 pip 来安装：
```bash
pip install jieba
```

使用 `jieba` 进行词性标注

`jieba` 使用 `jieba.posseg` 模块来进行词性标注。以下是一个简单的示例：

```python
import jieba.posseg as pseg

text = "词性标注是指为文本中的每个词分配一个词性"
words = pseg.cut(text)

for word, flag in words:
    print(f'{word}/{flag}', end=' ')
'''词性/n 标注/v 是/v 指为/v 文本/n 中/f 的/uj 每个/r 词/n 分配/vn 一个/m 词性/n '''
```

这段代码将对给定的文本进行分词和词性标注。**`pseg.cut` 函数返回一个迭代器，其中每个元素是一个 `pair` 对象，包含词语及其词性**。在上面的例子中，它会打印出每个词及其对应的词性。

关于词性标注

`jieba` 的词性标注是基于 `jieba` 自己的词性标注集来进行的，可能与其他标准略有不同。常见的词性包括：

- `n`：名词
- `v`：动词
- `a`：形容词
- `r`：代词
- `ns` ：地名
- `f`：方位名词
- `uj`：助词（在 `jieba` 中，`uj` 通常用于表示结构助词“的”）
- `vn`：名动词（动词性质的名词）
- `m`：数量词

#### hanlp词性标注[略]

```python
import hanlp
tagger = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ALBERT_BASE)
text = "汉克斯出生于加州的康科德市，他的父亲是厨师，母亲是医院工作者。"
pos_tags = tagger(text)
print(pos_tags)
```

结果类似于:

```python
# Output: [('汉克斯', 'NR'), ('出生', 'VV'), ('于', 'P'), ('加州', 'NR'), ('的', 'DEG'), ('康科德', 'NR'), ('市', 'NN'), ('，', 'PU'), ('他', 'PN'), ('的', 'DEG'), ('父亲', 'NN'), ('是', 'VC'), ('厨师', 'NN'), ('，', 'PU'), ('母亲', 'NN'), ('是', 'VC'), ('医院', 'NN'), ('工作者', 'NN'), ('。', 'PU')]
```

- `NR`: 名词-专有名词（Proper Noun）
- `VV`: 动词（Verb）
- `P`: 介词（Preposition）
- `DEG`: 关联词-的（Associative Particle）
- `NN`: 名词（Noun）
- `PU`: 标点符号（Punctuation）
- `PN`: 代词（Pronoun）
- `VC`: 动词-是（Verb-Copula）

### 文本的表示方法

#### 独热编码

在自然语言处理（NLP）中，**"One-Hot" 编码**（也称为独热编码）是一种表示分类变量的常用方法。在这种编码方式中，每个单词被表示为一个很长的向量。这个向量的长度等于词汇表的大小，其中每个单词被分配一个唯一的索引。在表示一个特定的单词时，其对应索引的位置为 1，而其他位置为 0。

这种表示方法的一个主要优点是它能**清楚地区分不同的单词**。但它的缺点是**向量的长度通常很长**，而且这种表示方法**不包含单词之间的任何关系信息**（例如，语义上相近的单词在独热编码中可能看起来完全不相关）。

下面我将使用 Python 的 `sklearn` 库（其中包括 `OneHotEncoder` 类）来演示如何对一组简单的词语进行独热编码，并对结果进行说明。

**示例**

假设我们有一个包含三个单词的简单词汇表：`["apple", "banana", "cherry"]`。

```python
from sklearn.preprocessing import OneHotEncoder 

# 创建词汇表
vocab = [["apple"], ["banana"], ["cherry"]]

# 初始化 OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# 对词汇表进行独热编码
one_hot_encoded = encoder.fit_transform(vocab)

# 打印结果
print("One-Hot Encoded Vocab:")
print(one_hot_encoded)
'''[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]'''

# 对新的词进行编码
new_word = [["banana"]]
new_word_encoded = encoder.transform(new_word)

print("\nEncoded New Word (banana):")
print(new_word_encoded)
'''[[0. 1. 0.]]
'''
```

**预期结果**

1. **词汇表的独热编码**：
   每个单词都会被转换成一个长度为 3 的向量（因为词汇表中有 3 个不同的单词），其中对应单词的位置为 1，其余位置为 0。

2. **对新词的编码**：
   当对词汇表之外的新词进行编码时，只有当这个新词存在于原先的词汇表中，它才能被正确编码。例如，对于单词 "banana"，将返回一个向量 `[0, 1, 0]`，表示它是词汇表中的第二个单词。

**为什么词汇表是字符串数组套数组**

在这个示例中，词汇表被构造为一个二维数组，每个内部数组包含一个单词。这是因为 `OneHotEncoder` 预期输入为二维数组，其中每一行代表一个样本，每一列代表一个特征。虽然在这个简单的例子中，每个样本（单词）只有一个特征（单词本身），但它仍需要作为二维数组提供。

假设我们有一个数据集，其中每个样本包含两个特征：水果种类和颜色。例如，我们的样本可以是这样的：

```python
samples = [
    ["apple", "red"],
    ["banana", "yellow"],
    ["cherry", "red"]
]
```

这里，每个样本有两个特征：第一个特征是水果种类（"apple"、"banana"、"cherry"），第二个特征是颜色（"red"、"yellow"）。

当我们对这些样本使用 `OneHotEncoder` 时，每个特征都会被独立地进行独热编码。例如：

```python
encoder = OneHotEncoder(sparse=False)
one_hot_encoded = encoder.fit_transform(samples)
```

输出结果可能是这样的：

```
[
    [1, 0, 0, 1, 0],  # apple, red
    [0, 1, 0, 0, 1],  # banana, yellow
    [0, 0, 1, 1, 0]   # cherry, red
]
```

在这个例子中，前三个数字代表水果种类，后两个数字代表颜色。因此，对于 "apple, red"，编码结果是 `[1, 0, 0, 1, 0]`，表示 "apple"（第一个元素为 1）和 "red"（第四个元素为 1），**相当于apple独热码和red独热码的拼接**。

**思考：**
如果`vocab = [["apple", "banana", "cherry"]]`，词汇表的编码结果会是什么？

```python
[[1. 1. 1.]]
```

因为只有一个样本，包含三个特征。由于这些特征在词汇表中是唯一的，所以每个特征的独热编码都是 1。

**`sparse=False` 的意思**

当使用 `OneHotEncoder` 时

```python
encoder = OneHotEncoder(sparse=False)
```

默认得到的是一个稀疏矩阵（通常是 CSR 格式）。

**压缩稀疏行（CSR）**: 存储所有非零元素的值，以及这些值的行索引，列索引。

```python
 (0, 0)	1.0
  (1, 1)	1.0
  (2, 2)	1.0
```

如果您需要一个常规的 NumPy 数组，可以设置 `sparse=False`


#### 对新词的编码

最后这个操作是在展示如何使用已训练（`fit`）的 `OneHotEncoder` 对新词进行编码。这个过程不会改变已经学习到的编码方式，而是将新词映射到已存在的编码上。

例如，当我们对单词 "banana" 使用 `transform` 方法时，`OneHotEncoder` 会查找 "banana" 在之前学习到的词汇表中的位置，并返回其对应的独热编码向量。如果这个新词在词汇表中不存在，编码器将无法正确编码它。在这个例子中，"banana" 是词汇表的第二个词，所以它的独热编码是 `[0, 1, 0]`。

注:

```python
new_word = [["banana"],['apple']]#查看多个词的编码
'''[[0. 1. 0.]
 [1. 0. 0.]]'''
```

#### 什么是 CBOW

CBOW（Continuous Bag of Words）是一种用于自然语言处理的模型，特别是在词嵌入（word embedding）领域中。CBOW 的目标是根据上下文中的单词来预测目标单词。在这个模型中，上下文是指目标单词周围的单词。

CBOW 模型的核心思想是，给定一个词的**上下文**（即这个词前后的一些词），模型应该能够**预测出这个词**是什么。例如，在句子 "The cat sits _ the mat" 中，给定上下文 "The cat sits on the mat"，CBOW 模型的任务是预测缺失的词（在这个例子中可能是 "on"）。

#### CBOW 下的 Word2Vec【完型填空】

Word2Vec 是一种广泛使用的词嵌入技术。Word2Vec 有两种主要的架构：CBOW 和 Skip-gram。在 CBOW 架构下，Word2Vec 模型使**用周围的上下文单词（即多个输入词）来预测目标单词**（即中心词）。相比之下，Skip-gram 模型则是用**一个单词来预测它周围的上下文**。

**示例**

假设我们有一句话：“A dog barks at night”。这句话中没有重复的单词，我们的词汇表（vocab）将包含这些单词：`{"A", "dog", "barks", "at", "night"}`。

**滑动窗口分析**

在 CBOW 模型中，我们通常为每个目标单词定义一个上下文窗口（context window）。假设我们选择的窗口大小为 3（即目标词的前一个词和后一个词作为上下文）。那么对于我们的示例句子，滑动窗口将产生以下输入和输出：

- 输入：["A", "barks"]，输出："dog"
- 输入：["dog", "at"]，输出："barks"
- 输入：["barks", "night"]，输出："at"

#### 0.前情提要：手写Embedding层

```python
import torch
import torch.nn as nn

# 假设的词汇大小和嵌入维度
vocab_size = 10
embedding_dim = 5

# 随机初始化嵌入矩阵
embedding_matrix = torch.rand(vocab_size, embedding_dim)
print("embedding_matrix",embedding_matrix)
indexs=torch.tensor([[0,1],[2,3]])
print("用矩阵作为索引",embedding_matrix[indexs])
'''矩阵查询的索引可以是一个张量,此时会对这个张量中的每个元素进行查询,结果按照张良的形状拼接起来'''

# 手动实现嵌入查找
def manual_embedding_lookup(indices):
    return embedding_matrix[indices]

# 使用 nn.Embedding
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# 将 nn.Embedding 的权重设置为与手动嵌入相同的值
'''不需要计算梯度'''
with torch.no_grad():
    embedding_layer.weight = nn.Parameter(embedding_matrix)
'''张量赋值给weight就是需要nn.Parameter'''

print("embedding_layer",embedding_layer.weight)

# 生成随机索引
indices = torch.randint(0, vocab_size, (3,2))

print("indices",indices)

# 使用手动嵌入方法
manual_embeds = manual_embedding_lookup(indices)

# 使用 nn.Embedding
nn_embeds = embedding_layer(indices)

# 比较结果
print("Manual Embedding Result:\n", manual_embeds)
print("\nnn.Embedding Result:\n", nn_embeds)
print("\nAre the results equal? ", torch.all(manual_embeds == nn_embeds))

```

注：`indices = torch.randint(0, vocab_size, (3,2))`

- 第一个参数 `0` 是生成随机整数的下限（包含）。
- 第二个参数 `vocab_size` 是上限（不包含），表示生成的随机数将小于 `vocab_size`。
- 最后一个参数 `(3,2)` 指定了张量的形状。

**结论:**

`nn.Embedding` 层**维护了一个嵌入矩阵（有vocab_size行，embedding_dim列）**，其中**每一行代表词汇表中一个单词的嵌入向量**。当给定一个索引张量（`indices`）时，`nn.Embedding` 层会**对这个索引中的每个元素进行查找操作**。具体来说，它会在嵌入矩阵中找到与这些索引对应的行,然后按照输入张量的形状拼接成一个张量。

#### `embedding_dim` 的含义

`embedding_dim` 是一个重要的参数，在创建词嵌入（word embeddings）时使用。它指定了嵌入向量的维度，即**每个单词被表示为多少维的向量**。

- **维度数**：`embedding_dim` 表示每个单词的嵌入向量中的特征数量。例如，如果 `embedding_dim` 为 100，那么每个单词都会被表示为一个包含 100 个数值的向量。
- **信息捕捉**：较高的维度可以使模型有更多的能力来捕捉和区分不同单词之间的细微差别，但同时也会增加模型的计算复杂性和对数据的需求。

#### 确定 `embedding_dim`

确定 `embedding_dim` 的过程涉及到几个因素的权衡：

1. **任务复杂性**：对于复杂的 NLP 任务或大型的词汇表，可能需要较高维度的嵌入向量以捕捉丰富的语义信息。
2. **数据量**：如果有大量的训练数据，可以尝试使用较高维度的嵌入，因为有足够的数据来学习这些额外的特征。
3. **计算资源**：较高的维度需要更多的计算资源和训练时间。如果资源有限，可能需要选择较低的维度。
4. **经验和实验**：通常，`embedding_dim` 的选择也基于经验和实验。在实践中，常见的维度包括 50、100、200 和 300。实验和模型调优可以帮助找到最适合特定任务的维度。

#### CBOW 模型的操作

要使用 PyTorch 构建一个 CBOW 模型进行 Word2Vec 训练，我们需要先定义模型架构，然后准备训练数据，并进行训练。以下是构建这个模型的步骤：

##### 1. 导入必要的库

首先，我们需要导入 PyTorch 及相关库：

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

##### 2. 定义 CBOW 模型

我们将定义一个简单的 CBOW 模型：

```python
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).mean(dim=0)
        out = self.linear(embeds)
        log_probs = torch.log_softmax(out, dim=0)
        return log_probs
```

**为什么要取平均**

在 CBOW 模型中，取平均是一种简化的方法，用**于将上下文中多个单词的信息合并成一个单一的表示**。这样做的好处是模型**不需要关注上下文中单词的具体顺序**，只需捕捉它们的整体语义信息。然而，这也可能是一个缺点，因为某些情况下单词的具体顺序是很重要的。为了捕捉更复杂的上下文关系，可以使用更高级的模型，如 LSTM 或 **Transformer**。
在反向传播过程中，CBOW 模型中的所有可训练参数都会被更新。这些包括：

1. **嵌入矩阵的权重**：`self.embeddings` 中的权重。这些权重定义了每个单词的嵌入向量。
2. **线性层的权重和偏置**：`self.linear` 中的权重和偏置。这些参数定义了从嵌入空间到输出空间（logits）的线性映射。

![image-20240220130953987](.\assets\image-20240220130953987.png)

##### 3. 准备数据

接下来，我们需要准备训练数据。首先，我们创建一个词汇表，并将单词映射到整数索引：

```python
word_to_ix = {"A": 0, "dog": 1, "barks": 2, "at": 3, "night": 4}
ix_to_word = {ix: word for word, ix in word_to_ix.items()}
vocab_size = len(word_to_ix)
#构建了2个字典，分别从字母映射到数字，数字映射到字母
print("word_to_ix",word_to_ix)
print("ix_to_word",ix_to_word)
print("vocab_size",vocab_size)
'''
word_to_ix {'A': 0, 'dog': 1, 'barks': 2, 'at': 3, 'night': 4}
ix_to_word {0: 'A', 1: 'dog', 2: 'barks', 3: 'at', 4: 'night'}
vocab_size 5
'''

data = [
    (torch.tensor([word_to_ix["A"], word_to_ix["barks"]]), torch.tensor(word_to_ix["dog"])),
    (torch.tensor([word_to_ix["dog"], word_to_ix["at"]]), torch.tensor(word_to_ix["barks"])),
    (torch.tensor([word_to_ix["barks"], word_to_ix["night"]]), torch.tensor(word_to_ix["at"]))
]

print(data)
'''[(tensor([0, 2]), tensor(1)), (tensor([1, 3]), tensor(2)), (tensor([2, 4]), tensor(3))]'''
#一个数组，当中有三个元素，每个元素对应一个滑动窗口，tensor([0, 2])是输入，tensor(1)是预测的值
```

##### 4.训练模型

现在我们可以初始化模型并进行训练：

```python
# 设置超参数
embedding_dim = 10

# 实例化模型
model = CBOW(vocab_size, embedding_dim)

# 定义损失函数和优化器
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(100):
    total_loss = 0
    for context, target in data:
        if epoch == 99:
            print(context, target)

        # 步骤 1. 准备数据
        context_idxs = context

        # 步骤 2. 运行模型的前向传递
        log_probs = model(context_idxs)

        if epoch == 99:
            print(log_probs)

        # 步骤 3. 计算损失
        loss = loss_function(log_probs.view(1, -1), target.view(1))
		'''view(1, -1) 将 log_probs 调整为一个形状为 (1, n) 的张量，而 view(1) 将 target 调整为一个形状为 (1,) 的张量'''
        # 步骤 4. 反向传播并更新梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss}")
```

#### 5. 测试模型
训练完成后，您可以使用模型来获取词嵌入，或尝试对新的上下文进行预测。

```python
# 测试数据
test_context = torch.tensor([word_to_ix["A"], word_to_ix["barks"]])

# 使用模型进行预测
with torch.no_grad():
    log_probs = model(test_context)

# 获取概率最高的单词索引
predicted_word_idx = torch.argmax(log_probs).item()

# 将索引转换回单词
predicted_word = ix_to_word[predicted_word_idx]

print(f"Input context: ['A', 'barks']")
print(f"Predicted word: '{predicted_word}'")

```

![image-20240220131322729](.\assets\image-20240220131322729.png)


请注意，这个示例是一个非常简化的版本，只用于演示基本的 CBOW 模型结构。在实际应用中，您会需要更大的词汇表、更多的数据、更长时间的训练，以及可能的超参数调优。此外，为了提高模型的性能和准确性，通常会采用更复杂的技术，如负采样（Negative Sampling)。

##### 词向量的含义

在经过训练的 CBOW 模型中，**嵌入矩阵中的每一行代表一个单词的词向量**。这些词向量捕捉了单词之间的语义关系。例如，对于词汇表 `{"A": 0, "dog": 1, "barks": 2, "at": 3, "night": 4}`，嵌入矩阵的第 1 行（索引为 0）表示单词 "A" 的词向量，第 2 行（索引为 1）表示单词 "dog" 的词向量，依此类推。

- 对于**英文**来说，单词通常被视为基本的语言单位，因此在许多应用中，每个单词会被表示为一个整体的词向量。例如，“Apple”作为一个单词，通常会对应于词嵌入空间中的一个点（即一个词向量）。

- **中文**处理略微复杂，因为中文写作是基于字符的，且很多词由多个字符组成，字符本身也有一定的语义。在中文NLP中，两种表示方法都很常见：

1. **字符级处理（Character-level）**：在这种方法中，每个中文字符被视为基本单位，并分别表示为词向量。这种方法简单直接，易于实现，但可能无法完全捕捉由多个字符组成的词汇的语义信息。
2. **词级处理（Word-level）**：在这种方法中，将识别出的中文词作为基本单位，每个词对应一个词向量。这要求进行分词处理，即将句子分割成词的序列。这种方法更能有效地捕捉词汇的语义信息，但分词的准确性对模型的性能有直接影响。

- **EOS**：结束标记（End-of-Sentence）用于指示句子或文本段的结束。当模型在生成文本时遇到EOS标记，它通常会停止当前句子或段落的进一步生成。EOS标记对于训练序列生成模型是非常重要的，因为它们需要知道何时结束生成过程。

##### 为何意思相近的单词的词向量更接近

在 CBOW 模型中，词向量通过上下文信息进行训练。模型学习如何根据给定的上下文预测目标单词。因此，**如果两个单词经常出现在相似的上下文中，它们的词向量会变得更接近**。这是因为这些向量需要捕捉相似的上下文信息以进行准确的预测。

##### 词向量接近性的测量方法

词向量之间的接近性**通常**通过计算它们之间的**余弦相似度**来衡量。余弦相似度测量的是两个向量在方向上的相似程度，而不是在数值大小上的相似性。如果两个向量的方向非常接近，它们的余弦相似度接近于 1；如果它们的方向相反，则接近于 -1。词向量的余弦相似度计算公式

余弦相似度衡量了两个向量在方向上的相似程度，**计算公式**如下：

$$cosine\_similarity(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

其中 `A` 和 `B` 是两个向量，$A \cdot B$ 是它们的点积，`||A||` 和 `||B||` 是它们的欧几里得范数（即向量的长度）。

##### CBOW 模型的完形填空能力

**CBOW 模型能够进行完形填空式的词预测**，是因为它在训练过程中学习了根据上下文预测目标单词的能力。模型通过观察大量的文本数据，学会了哪些单词更有可能出现在特定的上下文中。因此，给定一个包含空缺单词的句子，CBOW 模型可以根据上下文中的其他单词来预测最有可能填入该空缺的单词。


#### skip-gram 下的 Word2Vec


在 Skip-gram 模型中，与 CBOW 模型相反，我们使用目标单词来预测其上下文中的单词。这意味着对于给定的目标单词，模型试图预测其周围的单词。Skip-gram 模型特别适合处理大型数据集，并且对于频繁出现和不常见的单词都表现良好。

**示例句子和词汇表**

句子：“A dog barks at night”
词汇表（vocab）：{"A", "dog", "barks", "at", "night"}

**滑动窗口分析**

在 Skip-gram 模型中，对于每个目标单词，我们会查看其周围的上下文单词。假设我们选择的窗口大小为 3（即考虑目标词前后各一个词作为上下文），那么对于我们的示例句子，滑动窗口将产生以下输入和输出：

- 输入："dog"，输出：["A", "barks"]
- 输入："barks"，输出：["dog", "at"]
- 输入："at"，输出：["barks", "night"]

要实现 Skip-gram 架构下的 Word2Vec 模型，我们需要对模型结构进行一些调整。在 Skip-gram 模型中，目标是使用一个中心词来预测其上下文中的单词。因此，与 CBOW 相比，Skip-gram 模型的**输入和输出是颠倒的**：我们输入一个单词，然后尝试预测它的上下文。

##### 0.数据准备

```python
import torch
import torch.nn as nn
import torch.optim as optim

word_to_ix = {"A": 0, "dog": 1, "barks": 2, "at": 3, "night": 4}
ix_to_word = {ix: word for word, ix in word_to_ix.items()}
vocab_size = len(word_to_ix)

# 设置超参数
embedding_dim = 10
```

以下是实现 Skip-gram 模型的步骤：

##### 1. 定义 Skip-gram 模型

我们将定义一个新的 PyTorch 模型，其结构适用于 Skip-gram：

```python
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, word):
        embed = self.embeddings(word)
        out = self.linear(embed)
        log_probs = torch.log_softmax(out, dim=0)
        return log_probs
```

![image-20240220235355813](.\assets\image-20240220235355813.png)

##### 2. 准备 Skip-gram 训练数据

Skip-gram 模型的训练数据需要调整为中心词到上下文词的映射：

```python
skipgram_data = [
    (torch.tensor(word_to_ix["dog"]), torch.tensor([word_to_ix["A"], word_to_ix["barks"]])),
    (torch.tensor(word_to_ix["barks"]), torch.tensor([word_to_ix["dog"], word_to_ix["at"]])),
    (torch.tensor(word_to_ix["at"]), torch.tensor([word_to_ix["barks"], word_to_ix["night"]]))
]
```

##### 3. 训练 Skip-gram 模型

训练过程类似，但要注意损失函数的应用。由于 Skip-gram 模型的每个输入词对应多个输出词，因此我们需要适当地调整损失函数的计算：

```python
# 实例化模型
skipgram_model = SkipGram(vocab_size, embedding_dim)

# 同样的优化器和损失函数
optimizer = optim.SGD(skipgram_model.parameters(), lr=0.001)
loss_function = nn.NLLLoss()

# 训练模型
for epoch in range(100):
    total_loss = 0
    for center_word, context_words in skipgram_data:
        for context_word in context_words:
            log_probs = skipgram_model(center_word)
            loss = loss_function(log_probs, context_word)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss}")
```

##### 4. 测试 Skip-gram 模型

使用训练好的模型进行预测：

```python
# 测试数据
test_word = torch.tensor([word_to_ix["dog"]])

# 使用模型进行预测
with torch.no_grad():
    log_probs = skipgram_model(test_word)

# 获取概率最高的单词索引
predicted_indices = torch.topk(log_probs, 2).indices
#返回 log_probs 中最高的两个值的索引,例如tensor([[0, 1]])

predicted_words = [ix_to_word[idx.item()] for idx in predicted_indices[0]]
'''item() 方法将一个只包含单个元素的张量转换为一个 Python 标量'''

print(f"Input word: 'dog'")
print(f"Predicted context words: {predicted_words}")
```

![image-20240220235206777](.\assets\image-20240220235206777.png)
请注意，这只是一个基础的 Skip-gram 模型实现，简化了非常多的细节，可能需要进一步调优和大量数据进行训练以获得好的结果。在实际应用中，可以使用更高级的技术，如负采样（Negative Sampling)，来提高训练效率和模型性能。

#### 调库实现word2vec

##### 下载

```shell
wget -c http://mattmahoney.net/dc/enwik9.zip -P data
```

![image-20240304143231860](.\assets\image-20240304143231860.png)

下载了一个data文件，进入之后将其中的文件解压

![image-20240304143059716](.\assets\image-20240304143059716.png)

查看开头的字符
```shell
head -10 data/enwik9
```

![image-20240304143242591](.\assets\image-20240304143242591.png)

去GitHub的fastTest下载wikifil.pl
![image-20240304144455891](.\assets\image-20240304144455891.png)

利用wikifil.pl将文本当中的html标签去除

```shell
perl wikifil.pl enwik9 > wikitext
```

![image-20240304145209197](.\assets\image-20240304145209197.png)

如果Linux没有安装过python或者pip,之后安装fasttext

```shell
sudo apt install python3
sudo apt install python-pip
pip install fasttext
```

##### 训练

要使用FastText库进行Skip-gram模型的Word2Vec训练，您首先需要安装FastText。如果您还没有安装，可以通过Python的包管理器pip来安装。在您的命令行中运行：

```bash
pip install fasttext
```

接下来，您可以使用以下Python脚本来训练模型。这个脚本假定您已经有了处理好的文本文件，即您的700MB的维基百科语料。

```python
import fasttext

# 定义训练文件的路径
training_file = './data/wikitext.txt'  # 请将这里的路径替换成您文件的实际路径

# 训练模型
model = fasttext.train_unsupervised(training_file, model='skipgram')

# 保存模型
model.save_model('./data/word2vec_skipgram_model.bin')
```

这段代码首先导入`fasttext`库，然后指定您的训练文件路径。`fasttext.train_unsupervised`函数用于训练模型，其中`model='skipgram'`参数指定使用Skip-gram模型。训练完成后，模型被保存到文件中，以便以后使用。

请注意，FastText的训练可能会占用较多的计算资源，并且根据您的数据集大小和计算机的性能，可能需要一些时间来完成。

此外，FastText提供了多种参数来调整训练，例如调整向量的维度、学习率、上下文窗口大小等。您可以根据需要调整这些参数以优化模型的性能和准确性。这些参数可以作为`train_unsupervised`函数的参数提供。例如：

```python
model = fasttext.train_unsupervised(
    training_file, 
    model='skipgram',
    dim=100,         # 词向量维度
    ws=5,            # 上下文窗口大小
    epoch=5,         # 迭代次数
    minCount=5       # 忽略总频率低于此值的所有单词
)
```

您可以根据自己的需求和计算资源来调整这些参数。

##### 测试

您已经成功训练了一个使用 FastText 的 Skip-gram 模型，并且保存了训练好的模型。接下来，您可以加载这个模型，然后使用它来获取单词的词向量以及找出与某个单词最相近的单词。以下是如何操作的步骤：

1. **加载模型**：使用 FastText 的 `load_model` 函数加载您已经保存的模型。

2. **获取词向量**：使用模型的 `get_word_vector` 方法来获取特定单词的词向量。

3. **找出相近的单词**：使用模型的 `get_nearest_neighbors` 方法来找出与给定单词最相近的单词。

下面是相应的代码示例：

```python
import fasttext

# 加载模型
model = fasttext.load_model('./data/word2vec_skipgram_model.bin')

# 获取单词的词向量
word = "example"  # 替换为您感兴趣的单词
word_vector = model.get_word_vector(word)
print(f"词向量（{word}）: {word_vector}")

# 找出与特定单词最接近的单词
nearest_neighbors = model.get_nearest_neighbors(word,5) # k表示返回的最近邻单词的数量
print(f"与单词（{word}）最接近的单词及其相似度:")
for neighbor in nearest_neighbors:
    similarity,similar_word = neighbor
    print(f"{similar_word}, 相似度: {similarity},词向量:{model.get_word_vector(similar_word)}")
```

这段代码首先加载了之前保存的模型。然后，它会获取您选择的单词（在这个例子中是"example"）的词向量，并打印出来。接下来，它找出与这个单词最接近的5个单词及其相似度，并将这些信息打印出来。

请注意，要成功执行这些操作，您需要确保模型文件 `word2vec_skipgram_model.bin` 在当前工作目录中，或者提供其完整路径。同时，您可以通过更换 `word` 变量的值来探索不同单词的词向量和相似单词。

#### 词向量可视化

要在 Windows 上使用 PyTorch 实现词向量的可视化，你首先需要确保已经安装了相关的库。这里的关键是使用 PyTorch 来生成或加载词向量，然后使用 matplotlib 进行可视化。对于高维词向量（如100维或更高），通常使用 t-SNE（t-Distributed Stochastic Neighbor Embedding）技术来降维至2维或3维，以便可视化。

首先，确保已经安装了所需的库。如果还没有安装，可以使用 pip 安装 PyTorch、matplotlib 和 scikit-learn（包含 t-SNE 实现）：

```bash
pip install torch matplotlib scikit-learn
```

接下来，你可以使用以下代码来生成随机的词向量，然后用 t-SNE 进行降维，并使用 matplotlib 进行可视化：

```python
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 生成随机词向量
num_words = 100  # 词汇量大小
embedding_dim = 50  # 词向量维度
word_embeddings = torch.randn(num_words, embedding_dim)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=0)
word_embeddings_2d = tsne.fit_transform(word_embeddings)

# 可视化
plt.figure(figsize=(10, 10))
for i in range(num_words):
    plt.scatter(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1])
    plt.annotate(f'word_{i}', xy=(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')
plt.show()
```

这段代码首先创建了一个随机的词向量矩阵，然后使用 t-SNE 对这些高维词向量进行降维，最后使用 matplotlib 将这些词向量在2维空间中可视化。每个点代表一个词向量，且标有对应的索引（如 word_0, word_1, ...）。

![image-20240304233633090](.\assets\image-20240304233633090.png)

1. **`n_components=2`**：
   - 在`TSNE`函数中使用，这个参数设置了嵌入空间的维度。
   - 在这里，`n_components=2`意味着t-SNE将数据降维到2维。这对于数据可视化来说是理想的，因为我们可以在二维平面上绘制和理解这些点。

2. **`random_state=0`**：
   - 同样在`TSNE`中使用，这个参数设置了随机数生成器的种子。
   - 通过设置一个固定的`random_state`，确保每次运行t-SNE时得到的结果是一致的。t-SNE是一种随机算法，不同的随机种子可能导致结果有所不同。

3. **`tsne.fit_transform`**：
   - 这是t-SNE算法的主要函数，用于执行降维。
   - 它接收高维数据（在您的案例中是词向量），并将其转换为低维表示（2维）。这使得可以在二维平面上可视化高维数据。

4. **`figsize=(10, 10)`**：
   - 这是在`matplotlib`中设置图形大小的参数。
   - `figsize=(10, 10)`设置了绘制的图形大小为10x10英寸。

5. **`plt.scatter`**：
   - 这是`matplotlib`库中的一个函数，用于在图表中创建散点图。
   - 它在二维空间中绘制点，每个点的位置由x和y坐标决定。

6. **`xy=(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1])` 和 `xytext=(5, 2)`**：
   - 这些参数用于`plt.annotate`函数，它在matplotlib图表中添加文本注释。
   - `xy`指定注释指向的点的位置（即词向量的降维坐标）。
   - `xytext`指定文本注释的位置。这里是相对于`xy`点的偏移。

7. **`textcoords='offset points', ha='right', va='bottom'`**：
   - 也是`plt.annotate`的参数，用于进一步定义文本注释的样式和对齐方式。
   - `textcoords='offset points'`表示`xytext`的坐标是相对于`xy`点的偏移量，单位是点。
   - `ha='right'`和`va='bottom'`分别设置水平对齐（horizontal alignment）和垂直对齐（vertical alignment）为右对齐和底对齐。

#### 标签数量分布分析

**训练集位置：**
可以直接下载

```json
./data/wxTextClassification/train.news.csv
./data/wxTextClassification/test.news.csv
```



![image-20240309113310190](.\assets\image-20240309113310190.png)

**训练集说明:**

​	数据集是中文微信消息，包括微信消息的Official Account Name，Title，News Url，Image Url，Report Content，label。Title是微信消息的标题，label是消息的真假标签（0是real消息，1是fake消息）。训练数据保存在train.news.csv，测试数据保存在test.news.csv。

![image-20240309113546252](.\assets\image-20240309113546252.png)

**csv格式**
每一行数据之间用英文逗号隔开，行和行之间用换行隔开,数据当中也有逗号，但是都是中文逗号。

![image-20240309113757533](.\assets\image-20240309113757533.png)

​	**统计处理**

```py
import pandas as pd

# Load the datasets
train_data_path = './data/wxTextClassification/train.news.csv'
test_data_path = './data/wxTextClassification/test.news.csv'

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

print(test_data)
print(test_data['label'])
print(test_data.columns)

print("the distribution of the labels")
print("训练集标签分布")
print(train_data['label'].value_counts())
print(len(train_data))
print("测试集标签分布")
print(test_data['label'].value_counts())
print(len(test_data))


```

**输出**

```shell
the distribution of the labels
训练集标签分布
0    7844
1    2743
Name: label, dtype: int64
10587
测试集标签分布
0    8659
1    1482
Name: label, dtype: int64
10141
```

**解释:**

- **`train_data`** 是一个**数据帧**(DataFrame)，DataFrame是一个二维的、表格型的数据结构，您可以将它想象成一个**Excel表。**

-  **`['label']`** 指的是从这个DataFrame中**选择名为 `'label'` 的列**。结果是一个Series（一种Pandas数据结构），Series在Pandas中则是一个一维的数组结构。您可以将它视为Excel表中的**单独一列或一行**。

- **`len(test_data)`**: 这个函数返回的是DataFrame中行的数量，也就是样本的总数。`len()`不会包含表头行。
- **test_data.columns：**表头行（即列名称）

![image-20240309121802216](.\assets\image-20240309121802216.png)

**画图：**代码解释略,看看就行。

```python
import matplotlib.pyplot as plt
import numpy as np

# 绘制训练集和测试集的标签分布柱状图
train_label_counts = train_data['label'].value_counts()
test_label_counts = test_data['label'].value_counts()

# 创建一个图和子图
fig, ax = plt.subplots()

# 柱状图的数据
labels = ['Real News', 'Fake News']
train_counts = [train_label_counts[0], train_label_counts[1]]
test_counts = [test_label_counts[0], test_label_counts[1]]

# 设置柱状图的位置和宽度
x = np.arange(len(labels))  # 标签位置
width = 0.35  # 柱状图的宽度

# 绘制柱状图
rects1 = ax.bar(x - width/2, train_counts, width, label='Train')
rects2 = ax.bar(x + width/2, test_counts, width, label='Test')

# 添加一些文本用于标签、标题和自定义x轴刻度标签等
ax.set_ylabel('Counts')
ax.set_title('Label distribution in Training and Testing Sets')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 为每个条形图添加一个文本标签
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# 显示图形
plt.show()
```

![image-20240309124249011](.\assets\image-20240309124249011.png)

#### 句子长度分布图

```py
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据集
train_data_path = './data/wxTextClassification/train.news.csv'
train_data = pd.read_csv(train_data_path)

# 计算每个标题的长度
train_data['Title_Length'] = train_data['Title'].apply(len)
print(train_data['Title_Length'])

# 绘制标题长度的柱状图
plt.figure(figsize=(10, 6))
plt.hist(train_data['Title_Length'], bins=range(0, max(train_data['Title_Length']) + 10, 10), edgecolor='black')
plt.title('Distribution of Title Lengths in Training Set')
plt.xlabel('Title Length')
plt.ylabel('Number of Samples')
plt.xticks(range(0, max(train_data['Title_Length']) + 10, 10))
plt.show()

```

![image-20240309125504448](.\assets\image-20240309125504448.png)

**解释:**

1. **`train_data['Title_Length'] = train_data['Title'].apply(len)`**：这行代码的确在`train_data`这个DataFrame中**添加了一个新的列**`Title_Length`。这个新列的每个元素是由`train_data['Title']`这一列的相应元素（即每个标题）的长度计算得到的。**`.apply(len)`是将Python内置的`len`函数应用于`train_data['Title']`的每个元素上。**

2. **`plt.title`**：这个函数用于给图表添加一个标题。

3. **`plt.xlabel`**：这个函数用于给图表的x轴添加一个标签。

4. **`plt.ylabel`**：这个函数用于给图表的y轴添加一个标签。

5. **`plt.show()`**：这个函数用于**显示整个图表**。在某些环境中（如Jupyter Notebook），图表可能会自动显示，但在其他环境中（如普通的Python脚本），则需要调用`plt.show()`来显式显示图表。

6. **`plt.hist`**：这个函数用于**绘制直方图**。在这个上下文中，它用于显示`train_data['Title_Length']`中的数据分布，根据指定的bins（区间）将数据分布可视化，其中range部分是一样的。

7. **`plt.xticks`**：这个函数用于设置**x轴的刻度标签**。在您的代码中，它被用来设置x轴上的刻度，以便显示不同的标题长度区间。通过`range(0, max(train_data['Title_Length']) + 10, 10)`，**设置了从0开始，到最长标题长度加10，每隔10个单位一个刻度的x轴刻度。**

这些代码行共同工作，以生成一个直观的图表，展示了训练集中新闻标题长度的分布情况。
测试集：

####   词汇总数统计

 ```python
 import pandas as pd
 import jieba
 
 # 加载数据
 train_data_path = './data/wxTextClassification/train.news.csv'
 test_data_path = './data/wxTextClassification/test.news.csv'
 train_data = pd.read_csv(train_data_path)
 test_data = pd.read_csv(test_data_path)
 
 # 选择需要分词的列，例如：'Report Content'
 train_texts = train_data['Report Content'].tolist()
 test_texts = test_data['Report Content'].tolist()
 
 print(type(train_texts))
 print(train_texts[:5])#前五个元素
 
 # 分词函数
 def segment_words(texts):
     word_set = set()
     for text in texts:
         if isinstance(text, str):  # 确保文本是字符串
             words = jieba.lcut(text)
             word_set.update(words)
     return word_set
 
 # 统计训练集和测试集中的不同词语
 train_words = segment_words(train_texts)
 test_words = segment_words(test_texts)
 
 # 输出不同词语的数量
 print("Number of unique words in training set:", len(train_words))
 print("Number of unique words in testing set:", len(test_words))
 
 ```

**`tolist()`**: 这是Pandas Series的一个方法，用于将DataFrame的列转换为Python列表（list）。

**·isinstance()**: 这是一个内置的Python函数，用于检查一个对象是否是一个已知的类型。

**`lcut`**方法是`jieba`中用于进行分词的方法之一,例如，`jieba.lcut("我爱自然语言处理")`可能会返回`["我", "爱", "自然语言", "处理"]`。

**`update()`**: 这是Python集合（set）的一个方法。它用于将一个列表（或任何可迭代对象）中的元素添加到集合中。不同于列表，集合不包含重复元素。

```shell
<class 'list'>
['内容不符', '满口胡言', '？ ', '领个屁证，过你妹的七夕，几天前的图在今天拿来博眼球', '事件不实。']
Number of unique words in training set: 16157
Number of unique words in testing set: 18976
```

#### 词云统计

**环境：**

下载安装词云包，非常简单：

```bash
pip install wordcloud
```

![image-20240309215004108](.\assets\image-20240309215004108.png)

中文词云需要字体包,但是实际上Windows电脑自带了所有字体。

```json
C:\Windows\Fonts
```

![image-20240309220127849](.\assets\image-20240309220127849.png)

复制到data/fonts，可以发现三种，分别是常规（msyh.ttc）、粗体（msyhbd.ttc）和轻体（msyhl.ttc）。与`.ttf`（TrueType Font）文件不同，`.ttc`（TrueType Collection）文件是一个容器，内部可以包含多个风格相近的字体变种。在大多数情况下，`.ttc`文件可以直接用于大部分需要字体的应用，包括`wordcloud`库。

![image-20240309220753076](.\assets\image-20240309220753076.png)


**代码：**

```python
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba

# 加载数据
train_data_path = './data/wxTextClassification/train.news.csv'
train_data = pd.read_csv(train_data_path)

# 定义生成词云的函数
def generate_wordcloud(text, font_path):
    word_list = jieba.lcut(text)
    clean_text = ' '.join(word_list)
    wordcloud = WordCloud(font_path=font_path, width=800, height=800, background_color='white').generate(clean_text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

print(train_data['label'] == 0)#对于每一行，如果label等于0，返回True，否则返回False
print(train_data[train_data['label'] == 0])#对于每一行，如果刚才的结果为True，保留，否则删除
print(train_data[train_data['label'] == 0]['Report Content'])#对上面的结果，只要Report Content这一列，也就是所有真新闻的内容

# 真新闻和假新闻的内容
real_news_content = ' '.join(train_data[train_data['label'] == 0]['Report Content'].dropna())
fake_news_content = ' '.join(train_data[train_data['label'] == 1]['Report Content'].dropna())

# 替换为您的中文字体路径
font_path = './data/msyh.ttc'

# 生成并显示真新闻的词云
print("真新闻词云：")
generate_wordcloud(real_news_content, font_path)

# 生成并显示假新闻的词云
print("假新闻词云：")
generate_wordcloud(fake_news_content, font_path)

```

可以看到假新闻词云：
![image-20240309221315384](.\assets\image-20240309221315384.png)

- `train_data['label'] == 0`: 这个条件用于选择`train_data` DataFrame中`label`列值为0的行，即真新闻。
- `train_data[...]['Report Content']`: 这个部分获取符合上述条件的行的`Report Content`列。
- `.dropna()`: `dropna`方法用于移除所有包含缺失值（NaN）的行，以防止在后续的处理中引发错误。
- `' '.join(...)`: 这个函数将一个列表（或任何可迭代的序列）中的元素合并成一个单独的字符串，元素之间以空格分隔。
- `jieba.lcut`函数将文本字符串`text`分割成一个包含词语的列表。所以，`word_list`的类型是`list`。
- `clean_text`的类型是`str`，即字符串。元素之间以空格`' '`分隔
- `figsize`: 以英寸为单位的图形大小，此处指定为8x8英寸的正方形。
- `facecolor`: 图形的背景颜色，默认为`None`，这意味着它将使用matplotlib的默认设置。
- **`plt.axis("off")`**: 关闭坐标轴，不显示坐标轴和边框。
- `tight_layout`：自动调整子图参数，以给定的填充部分适配区域。
- `pad=0`: 指定了子图周围的填充区域的大小。这里设置为0，意味着没有额外的空间。

#### n-gram

N-gram是文本处理中的一个概念，它是从文本中提取的一系列连续的N个项目（例如，字母、音节或单词）。N-gram模型是基于一个假设：一个项目的出现依赖于它前面的N-1个项目。这是一种语言模型，通常用于自然语言处理任务中，例如拼写检查、语音识别、文本生成等。

- **Unigram (1-gram)**: **每个项目单独出现**，不考虑邻近的项目。例如，句子"The quick brown fox"的unigram是"the"、"quick"、"brown"、"fox"。

- **Bigram (2-gram)**: **两个连续项目的序列**。上面句子的bigram序列是"the quick"、"quick brown"、"brown fox"。Bigram模型考虑一个项目出现与它前面一个项目的关系。

- **Trigram (3-gram)**: **三个连续项目的序列**。同样句子的trigram序列是"the quick brown"、"quick brown fox"。Trigram模型则是基于前面两个项目来预测第三个项目。

N-gram模型的N值可以更大，但随着N的增大，计算量和数据稀疏性问题会迅速增加。因此，bi-gram和tri-gram是比较**常用的**N-gram模型。在处理自然语言时，bi-grams和tri-grams可以捕捉到更多的语言结构信息，从而帮助理解和预测文本。









### RNN

#### 前置知识:联合概率

联合概率（Joint Probability）是指在给定某一情况下，两个（或多个）事件同时发生的概率。在语言模型的上下文中，联合概率$ p(x_1, x_2, x_3, \ldots) $指的是序列中所有单词按特定顺序出现的概率。这个概率反映了一个句子或文本序列出现的可能性。

**举例说明**

假设我们有一个简单的句子：“The cat sits on the mat。”**语言模型的任务是预测这个句子出现的联合概率**，即：

$$ p(\text{"The"}, \text{"cat"}, \text{"sits"}, \text{"on"}, \text{"the"}, \text{"mat"}) $$

**这个联合概率是指整个序列“the cat sits on the mat”作为一个整体出现的概率**。在实践中，直接计算整个序列的联合概率是非常困难的，因为需要考虑所有可能的单词组合及其出现的频率。因此，语言模型通常会利用**链式法则（**Chain Rule）来简化计算，将联合概率分解为条件概率的乘积：

$$ p(\text{"The"}, \text{"cat"}, \text{"sits"}, \text{"on"}, \text{"the"}, \text{"mat"}) = p(\text{"The"}) \times p(\text{"cat"} | \text{"The"}) \times p(\text{"sits"} | \text{"The cat"}) \times \ldots $$

- $ p(\text{"The"}) $是句子以“The”开始的概率。
- $ p(\text{"cat"} | \text{"The"}) $是在“The”之后出现“cat”的条件概率。
- $ p(\text{"sits"} | \text{"The cat"}) $是在“The cat”之后出现“sits”的条件概率。
- 依此类推，直到整个序列。

通过这种方式，语言模型可以通过学习文本数据中的单词序列来估计这些条件概率，进而能够评估任意文本序列出现的可能性，或者预测下一个单词是什么。

**顺序**
联合概率内含着顺序性，这意味着序列中单词出现的顺序对于计算联合概率是至关重要的。因此，"The cat sits on the mat"的联合概率与"The mat sits on the cat"的联合概率是不同的，**这两个序列的概率密度计算公式也会有所不同，正是这种差异体现了顺序的重要性。**

对于序列1 ("The cat sits on the mat")，其联合概率可以分解为：

$$ p(\text{"The"}, \text{"cat"}, \text{"sits"}, \text{"on"}, \text{"the"}, \text{"mat"}) = p(\text{"The"}) \times p(\text{"cat"} | \text{"The"}) \times p(\text{"sits"} | \text{"The cat"}) \times \ldots $$

而对于序列2 ("The mat sits on the cat")，其联合概率分解为：

$$ p(\text{"The"}, \text{"mat"}, \text{"sits"}, \text{"on"}, \text{"the"}, \text{"cat"}) = p(\text{"The"}) \times p(\text{"mat"} | \text{"The"}) \times p(\text{"sits"} | \text{"The mat"}) \times \ldots $$

每个序列的条件概率是基于前面单词序列的具体排列计算的，因此两个序列的概率计算是不同的。语言模型学习的目的就是要准确估计这种条件概率，从而能够理解和生成语言结构上合理且语义上连贯的句子。

**单词概率的频率估计**

利用频率估计概率是统计和机器学习中常见的做法，尤其是在处理自然语言处理（NLP）问题时。基本思想是使用某个事件出现的频率作为该事件概率的估计。在语言模型的上下文中，这意味着可以通过统计单词或单词序列出现的次数来估计它们出现的概率。

单个单词的概率可以通过该单词出现的次数除以所有单词出现的总次数来估计。例如，单词"The"的概率可以估计为：

$$ p(\text{"The"}) = \frac{n(\text{"The"})}{n(\text{"total words"})} $$

其中，$n(\text{"The"})$是单词"The"在语料库中出现的次数，而$n(\text{"total words"})$是语料库中所有单词出现的总次数。

**条件概率的频率估计**

条件概率，如$p(\text{"cat"} | \text{"The"})$，可以通过统计两个单词连续出现的次数与前一个单词出现的次数之比来估计。例如：

$$ p(\text{"cat"} | \text{"The"}) = \frac{n(\text{"The cat"})}{n(\text{"The"})} $$

这里，$n(\text{"The cat"})$**表示序列"The cat"在语料库中出现的次数**，$n(\text{"The"})$是单词"The"出现的次数。

**更长序列的条件概率**

对于更长的序列，如$p(\text{"sits"} | \text{"The cat"})$，估计方法类似，只不过分子是更长序列的出现次数，分母是不包括最后一个单词的序列出现次数：

$$ p(\text{"sits"} | \text{"The cat"}) = \frac{n(\text{"The cat sits"})}{n(\text{"The cat"})} $$


**注意事项**

- **数据稀疏问题**：当处理大型语料库时，很多单词序列可能很少出现或根本不出现。这会导致频率估计的概率为零，影响模型的性能。为了解决这个问题，可以采用平滑技术，如加一平滑（Laplace smoothing）。
- **效率**：对于大型语料库，直接计算长序列的条件概率可能非常耗时和存储密集。因此，实际应用中常常采用更高级的模型和技术，如N-gram模型、神经网络语言模型等。

通过这种方式，我们可以使用频率来近似计算出现在文本序列中各种单词组合的概率，从而构建语言模型。

![](.\assets\image-20240314205028670.png)


#### N元语法

当我们在处理语言模型时，马尔可夫假设能够大大简化联合概率的计算。这个假设基于一个简单的原则：一个词的出现只依赖于前面有限个词的序列。这就允许我们使用N-gram模型来近似语言模型中的联合概率，其中N-gram模型是基于前$N-1$个词来预测第$N$个词出现的概率模型。

**一元语法（Unigram）**

一元语法模型是最简单的N-gram模型，其中每个词的出现被假设为独立的。这意味着**联合概率可以简化为单词概率的乘积**，不考虑词与词之间的依赖关系：

$$ p(x_1, x_2, x_3, \ldots) \approx p(x_1) \times p(x_2) \times p(x_3) \times \ldots  \approx \frac{n(x_1)}{N} \times \frac{n(x_2)}{N} \times \frac{n(x_3)}{N} \times \ldots $$

这种模型忽略了单词之间的上下文关系，因而在捕捉语言的连贯性和上下文方面表现不佳。

**二元语法（Bigram）**

**二元语法模型考虑每个词的出现依赖于它前面的一个词**。这种模型通过计算条件概率来近似联合概率，从而在一定程度上捕捉了词与词之间的关系：

$$ p(x_1, x_2, x_3, \ldots) \approx p(x_1) \times p(x_2 | x_1) \times p(x_3 | x_2) \times \ldots \approx \frac{n(x_1)}{N} \times \frac{n(x_1, x_2)}{n(x_1)} \times \frac{n(x_2, x_3)}{n(x_2)} \times \ldots  $$

二元语法模型比一元语法模型更能捕捉文本的局部上下文信息，但仍然限于短范围内的依赖。

**三元语法（Trigram）**

三元语法模型扩展了二元语法模型的概念，**它考虑每个词的出现依赖于它前面的两个词**。这使得模型能够更好地捕捉更长范围内的上下文依赖：

$$ p(x_1, x_2, x_3, \ldots) \approx p(x_1) \times p(x_2 | x_1) \times p(x_3 | x_1, x_2) \times \ldots \approx \frac{n(x_1)}{N} \times \frac{n(x_1, x_2)}{n(x_1)} \times \frac{n(x_1, x_2, x_3)}{n(x_1, x_2)} \times \ldots $$

三元语法模型在捕捉上下文和连贯性方面比一元和二元语法模型有更好的表现，但计算和存储需求也相应更高。

**优点**

- **处理未见词组合**：能够为未在训练集中出现过的词组合分配非零概率。这使得模型在面对新的文本或少见的词组时具有一定的鲁棒性。

**缺点**

- **空间复杂度**：对于N元语法模型，假设语料库中有$V$个唯一词汇。那么，理论上模型需要存储的**不同的N-gram的数量最多为**$V^N$。因此，空间复杂度为$O(V^N)$。这意味着空间需求随着词汇量的增加和N的增大而指数级增长，导致巨大的存储需求。
    - 例如，对于一个拥有10,000个唯一单词的语言（$V=10,000$）和一个三元语法模型（$N=3$），理论上可能有$10,000^3 = 10^{12}$个不同的三元组(例如"The cat on","cat cat cat"...)需要被存储和处理。实际上，并非所有可能的组合都会出现，如果只存储训练文本当中出现的N元组，只需要把总词数为n的文本扫描一遍。
    - 可以设置一个最小频率或概率阈值，仅保留那些**频率高于此阈值的N-gram**，对于那些在训练集中很少出现或者根本未出现的N-gram，可以将它们从模型中移除。这种做法可以显著减少模型的大小，同时对模型性能的影响较小，因为那些低频的N-gram对于模型的贡献通常较小。

**结论：通常使用二三元语法。**

### RNN的可视化和手敲实现

#### RNN的输入

**RNN**: Recurrent Neural Network（循环神经网络）

在循环神经网络（RNN）中，$x_t$（其中$t$是下标）**表示在时间步$t$的输入**。RNN是专门为了处理序列数据设计的神经网络模型，它能够通过**维护一个内部状态（或称为“隐藏状态”）来捕获序列中的时间动态特征。**序列可以是文字、时间序列数据、音频信号等等，每个时间步$t$都对应序列中的一个元素。

RNN中的时间步$t$

在时间步$t$，$x_t$就是该时间点对应的输入。例如：

- **在文本处理应用中，$x_t$可能是在时间步$t$的单词或字符的独热编码（one-hot encoding）或词嵌入向量（word embedding vector）。**
- 如果我们有一个序列“你好世界”，那么：

  - $x_1$ 对应于“你”,
  - $x_2$ 对应于“好”,
  - $x_3$ 对应于“世”,
  - $x_4$ 对应于“界”。
- **在股票价格预测应用中，$x_t$可能是第$t$天的股票价格或其他相关的财经指标。**
- **在语音识别应用中，$x_t$可能是时间步$t$的音频信号的特征向量。**

![image-20240315165431449](./assets/image-20240315165431449.png)

#### $h_t$的含义

**每个$h_t$代表了截至当前时间步$t$，模型对之前输入序列的“记忆”或“理解”。**例如： $h_3$：表示在处理完$x_1, x_2, x_3$之后，模型的内部状态【在文本类任务当中，是对前3个词的""理解""】。

- 如果我们有一个序列“你好世界”，那么：
  - $h_1$ 反映了模型对“你”的理解，
  - $h_2$ 结合了$h_1$（即对“你”的理解）和新的输入$x_2$（即“好”），从而反映了模型对“你好”的理解，
  - $h_3$ 进一步结合了$h_2$和新的输入$x_3$（即“世”），从而反映了模型对“你好世”的理解，
  - $h_4$ 则结合了$h_3$和新的输入$x_4$（即“界”），从而反映了模型对整个序列“你好世界”的综合理解。

**在每个时间步$t$，RNN会根据当前的输入$x_t$和前一时间步的隐藏状态$h_{t-1}$来计算当前的隐藏状态$h_t$。**每个隐藏状态$h_t$不仅取决于当前的输入$x_t$，也依赖于前一个隐藏状态$h_{t-1}$，这反映了序列的连续性和上下文依赖性。
这个过程可以通过下面的公式概括：

$$ h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b_h) $$

其中，$f$是激活函数，$W_{xh}$是输入到隐藏状态的权重矩阵，$W_{hh}$是隐藏状态到隐藏状态的权重矩阵，$b_h$是隐藏状态的偏置项【命名上很有规律】。

**现实意义：$h_t$表示模型对前t个输入的理解，本公式意味着模型可以在前`t-1`个输入的理解的基础上，加上当前的输入，得到对所有`t`个输入的理解。有点动态规划的感觉。就类似于基于我对背景故事的理解，再加上当前章节去得到对整个故事的理解。**

RNN可以通过这种递归方式不断更新其隐藏状态，从而在处理序列数据时考虑到之前的信息。这使得RNN非常适合处理那些输出依赖于先前输入的任务，如语言建模、序列生成、时间序列预测等。

#### $h_0$

![image-20240315175445065](./assets/image-20240315175445065.png)

当计算$h_1$，即序列的第一个隐藏状态时，理论上需要用到$h_0$和$x_1$。

在循环神经网络（RNN）中，$h_0$通常被视为序列处理之前的初始隐藏状态。$h_0$**通常初始化为一个全零向量，意味着在开始处理序列之前，没有任何之前的信息或“记忆”。**这是最常见的初始化方式，尤其是在处理独立的序列或句子时。
在一些特殊应用中，如处理一个长文本或连续的数据流时，序列之间的$h_t$可能会被传递和使用。这意味着，一个序列的最终隐藏状态$h_T$（其中$T$是序列的最后一个时间步）可以作为下一个序列开始时的初始状态$h_0$。

#### $y_t$的含义

在循环神经网络（RNN）中，除了计算当前时间步的隐藏状态$h_t$之外，模型通常还会计算一个输出$y_t$。这个输出代表了**在时间步$t$，给定到目前为止的输入序列（直到$x_t$），网络的最终输出或预测。**

- 在不同的应用场景中，$y_t$可以有不同的含义。例如，**在文本生成任务中，$y_t$可能代表下一个单词的概率分布；在股票价格预测中，它可能代表下一个时间步的价格预测；在情感分析中，$y_t$可能代表句子的情感标签的概率分布。**
- $y_t$是根据当前时间步的隐藏状态$h_t$（以及有时当前的输入$x_t$）来计算的，这反映了RNN模型对当前时刻及之前时刻信息的综合理解
- 如果我们有一个序列“你好世界”，那么：
  如果我们使用RNN进行序列生成或预测任务，$y_4$可以被视为基于当前模型理解（即到“界”为止的理解）对序列中下一个元素（即第五个字）的预测。

$y_t$的计算通常涉及将当前时间步的隐藏状态$h_t$【这里没有t-1】通过一个或多个线性层（也叫全连接层）并应用一个激活函数。具体来说，计算公式可能如下所示：

$$ y_t = g(W_{hy}h_t + b_y) $$

其中：
- $W_{hy}$是从隐藏状态到输出层的权重矩阵。
- $b_y$是输出层的偏置项【命名上很有规律】。
- $g$是激活函数，它的选择取决于具体任务。例如，在分类任务中常用的是softmax函数，它可以将输出转换为概率分布。

**现实意义：本公式意味着可以用模型对`t`个输入的理解得到一个预测(反馈)，就类似于基于我对数学的理解（`h_t`），去对数学题给出一个答案(`y_t`)。**

#### $y_t$激活函数的选择

- **语言模型**：在字符级语言模型中，$y_t$可以是对下一个字符的概率分布的预测。使用**softmax激活函数**，可以确保所有可能字符的预测概率加起来等于1。
- **时间序列预测**：在时间序列预测中，如果目标是预测下一个时间点的值，$y_t$可能是一个线性激活函数的结果，表示预测值。

#### 张量形状分析

在循环神经网络（RNN）中，给定输入$x_t$是一个具有10000个元素的向量，而隐藏状态$h_t$是一个具有200个元素的向量，我们可以分析权重矩阵$W_{xh}$、$W_{hh}$和偏置项$b_h$的形状，以及如何通过这些组件计算$h_t$。

**权重矩阵和偏置项的形状**

1. **$W_{xh}$**：这是从输入层到隐藏层的权重矩阵。由于它需要将10000维的输入转换为200维的隐藏状态，$W_{xh}$的形状必须是$200 \times 10000$。

2. **$W_{hh}$**：这是从前一个隐藏状态到当前隐藏状态的权重矩阵。为了将200维的前一个隐藏状态$h_{t-1}$转换为200维的当前隐藏状态$h_t$，$W_{hh}$的形状必须是$200 \times 200$。

3. **$b_h$**：这是隐藏层的偏置项。由于隐藏状态是200维的，所以$b_h$的形状必须是$200$。

**$h_t$的计算**

给定上述形状的权重矩阵和偏置项，$h_t$的计算可以通过下面的公式实现：

$$ h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b_h) $$

**张量形状分析**

- $W_{xh}x_t$的结果是一个$200$维的向量（$200 \times 10000$乘以$10000 \times 1$）。
- $W_{hh}h_{t-1}$的结果也是一个$200$维的向量（$200 \times 200$乘以$200 \times 1$）。
- 将这两个结果向量相加，并加上偏置项$b_h$（也是一个$200$维的向量），最终结果仍然是一个$200$维的向量。
- 应用激活函数$f$后，$h_t$的形状不变，仍为$200$（或视为$200 \times 1$），这就是新的隐藏状态。

#### 简化公式

![image-20240317214908262](./assets/image-20240317214908262.png)

对于循环神经网络（RNN）中的计算过程，我们可以通过合并权重矩阵和输入向量来简化计算。这种方法利用了矩阵运算的性质，可以减少计算步骤并提高效率。

原始的计算公式是：

$$ h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b_h) $$

为了简化这个计算，**我们可以将输入$x_t$和前一个隐藏状态$h_{t-1}$竖直拼接（堆叠）成一个新的大向量，同时将权重矩阵$W_{xh}$和$W_{hh}$水平拼接（并排）成一个新的大权重矩阵**。这样，上述计算可以被重新表述为：

$$ h_t = f(W_h [x_t; h_{t-1}] + b_h) $$

这里，$W_h$是拼接后的权重矩阵，$[x_t; h_{t-1}]$是将$x_t$和$h_{t-1}$拼接而成的大向量。

**张量形状分析（基于原始数据）**

给定：
- $x_t$是一个10000维的向量，形状为$10000 \times 1$。
- $h_{t-1}$是一个200维的向量，形状为$200 \times 1$。
- $W_{xh}$的形状为$200 \times 10000$。
- $W_{hh}$的形状为$200 \times 200$。
- $b_h$是一个200维的向量。
- ![image-20240317215312513](./assets/image-20240317215312513.png)

**拼接后的形状：**

- 当我们拼接$x_t$和$h_{t-1}$，新的向量$[x_t; h_{t-1}]$的形状为$(10000 + 200) \times 1 = 10200 \times 1$。
- 当我们水平拼接$W_{xh}$和$W_{hh}$，新的权重矩阵$W_h$的形状为$200 \times (10000 + 200) = 200 \times 10200$。

**简化后的公式中张量形状：**

- $W_h [x_t; h_{t-1}]$的结果是一个$200 \times 10200$乘以$10200 \times 1$的矩阵乘法，得到的结果是一个$200 \times 1$的向量。
- 加上偏置$b_h$（$200 \times 1$的向量）后，结果仍然是一个$200 \times 1$的向量。
- 应用激活函数$f$之后，$h_t$的形状不变，为$200 \times 1$。

**补充：**

- $ h $：hidden state（隐藏状态）
- $ b $：bias（偏置）
- $ W $：weight（权重）



#### nn.RNN

**在PyTorch中，`nn.RNN`模块提供了一个简单的循环神经网络层的实现。你可以直接调用这个模块来构建RNN模型，而不必定义一个新的类**，尤其是在你想要快速原型或者演示基本概念时。这里我将介绍如何使用`nn.RNN`，包括初始化参数、准备输入数据以及前向传播。

**初始化`nn.RNN`**

当你初始化`nn.RNN`时，需要指定几个关键的参数：

- **`input_size`：输入特征的数量。**
- **`hidden_size`：RNN隐藏层的特征数量。**
- **`num_layers`（可选）：RNN的层数，默认为1。**
- 其他可选参数，如**`nonlinearity`（选择激活函数，默认是`'tanh'`）**，`batch_first`（默认为`False`，如果设置为`True`，则输入输出的数据格式为`(batch, seq, feature)`），等等。

**准备输入数据**

**RNN期望的输入是一个三维张量，其形状为`(seq_len, batch, input_size)`，除非你将`batch_first=True`传递给RNN构造函数，在这种情况下输入形状应该是`(batch, seq_len, input_size)`。**

示例：随机数据输入

以下是一个使用`nn.RNN`和随机生成的输入数据的例子。

假设我们想要处理一个序列长度为5的批次，每个序列有3个特征，并且我们设定隐藏层的大小为10。

<img src="./assets/image-20240317165528202.png" alt="image-20240317165528202" style="zoom:67%;" />

1. **初始化`nn.RNN`：**
   - `input_size=3`意味着每个时间步的**输入$x_t$的向量大小是3**。这意味着，无论你是在处理词向量、特征向量还是任何其他形式的输入数据，每个时间步的输入都应该是一个3维向量。
   - `hidden_size=10`表示隐藏状态**$h_t$的向量大小是10**。这决定了RNN中隐藏层的维度。每个时间步更新时，隐藏状态的计算都会产生一个10维的向量，这个向量携带了至当前时间步为止序列的信息。
   - `num_layers=1`指的是RNN网络堆叠的层数。**对于最简单的RNN，我们通常使用一层**，但是在复杂的任务中，可能会使用多层RNN堆叠起来，形成一个更深的模型。
```python
rnn = nn.RNN(input_size=3, hidden_size=10, num_layers=1, batch_first=True)
```

2. **生成随机输入数据：**
  - **批次大小（Batch Size）**: 第一个维度（在这个例子中是2）代表批次大小。这意味着我们**同时处理两个独立的序列**。在实际应用中，为了提高计算效率，我们通常会并行处理多个序列。

  - **序列长度（Sequence Length）**: 第二个维度（在这个例子中是5）代表序列的长度，或者说是每个序列中的时间步数量。这表示每个序列包含5个元素（**在文本处理中，可以是5个词**）。

  - **输入大小（Input Size）**: 第三个维度（在这个例子中是3）代表每个时间步的输入向量的大小。在文本处理中，如果我们将每个词表示为一个词向量，这里的3意味着**每个词由一个3维的向量表示。**

    <img src="./assets/image-20240317171058038.png" alt="image-20240317171058038" style="zoom:50%;" />
```python
input_data = torch.randn(2, 5, 3)  # (batch, seq_len, input_size)
```

- **单个时间步的输入**：在最简单的情况下，比如处理单个序列而非批量数据时，你可以想象每个时间步的输入$x_t$为一个词向量。如果输入是文本数据，每个词可以通过某种方式（如one-hot编码或词嵌入）被转换成一个向量。
- **多个时间步下的输入**：当考虑一个序列中多个时间步的输入时，可以将这些词向量（假设每个都是3维的）按序排列，形成一个矩阵，其形状为序列长度×输入大小（在这个例子中是5×3）。

- **批量处理的序列**：在批量处理中，为了同时处理多个这样的序列，我们将它们堆叠起来，形成一个三维张量。所以，这里的`input_data`实际上包含了2个序列，每个序列长度为5，每个时间步的输入大小为3，从而形成了一个形状为2×5×3的张量。

3. **前向传播：**
```python
output, hidden = rnn(input_data)
```

这里，`output`包含了最后一层每个时间步的输出特征$h_t$，`hidden`包含了来自网络最后一个时间步的隐藏状态。对于简单的RNN，`hidden`就是`output`的最后一个时间步的输出。

接下来，让我们执行这个示例代码。

```py
import torch
import torch.nn as nn

torch.manual_seed(42)
# 初始化nn.RNN
rnn = nn.RNN(input_size=3, hidden_size=10, num_layers=1, batch_first=True)

# 生成随机输入数据
input_data = torch.randn(2, 5, 3)  # (batch, seq_len, input_size)

# 前向传播
output, hidden = rnn(input_data)

# 查看输出形状
print(W_ih.shape, W_hh.shape, b_ih.shape, b_hh.shape)

```

执行示例代码后，我们得到了`output`和`hidden`的形状：

- `output`的形状是`torch.Size([2, 5, 10])`，这表示我们有2个序列（批次大小为2），序列长度为5，每个时间步的输出特征大小为10（隐藏层的大小）。

  1. **输入的数据有多少个序列，输出的数据也有多少个序列.**

     输入数据中的序列数量（在这个例子中是批次大小`batch_size=2`）直接决定了输出数据中序列的数量。这意味着，如果你输入了2个序列，RNN会分别对这两个序列进行处理，并为每个序列产生一个输出序列。因此，输出数据（`output`）中也会包含2个序列。

  2. **输入的数据有多少个时间步，输出的数据也有多少个时间步.**

     输入数据中每个序列的时间步数量（在这个例子中是`seq_len=5`）将与输出数据中每个序列的时间步数量相匹配。这意味着，如果每个输入序列包含5个时间步，RNN会在每个时间步产生一个输出，因此每个输出序列也会包含5个时间步。换句话说，RNN在每个时间步产生一个输出，这些输出一起形成了输出序列。

  3. **输出代表的是$h_t$，而不是$y_t$**

     `output`张量包含的是RNN在每个时间步产生的隐藏状态$h_t$的集合。在很多情况下，这些隐藏状态可以直接作为输出使用，尤其是当RNN被用于特定任务如特征提取时。然而，要注意的是，这些隐藏状态**$h_t$可以经过额外的处理**（如通过额外的网络层）来产生最终的预测输出$y_t$。在很多应用中，比如分类、回归或序列生成任务，我们可能会在RNN的基础上添加额外的层（例如全连接层）来将隐藏状态$h_t$转换为具体任务所需的输出格式$y_t$。

  简而言之，RNN的**`output`张量表示了每个时间步的隐藏状态$h_t$**，它们是模型在处理序列时每一步的内部表示。如果需要根据这些隐藏状态计算特定任务的输出$y_t$，通常需要在RNN之后添加额外的处理层。
- `hidden`的形状是`torch.Size([1, 2, 10])`，**这表示最后一个时间步的隐藏状态**，其中1表示RNN层的数量，2是批次大小，10是隐藏层的特征数量。






#### 纯手写RNN

为了手动计算循环神经网络（RNN）的输出并与PyTorch的实现进行比较，我们将首先提取出RNN层的权重和偏置参数，然后实现一个函数来模拟RNN的前向传播过程。最后，我们将比较这个手动实现的结果与PyTorch `nn.RNN`模块的输出。

**输入**

有一个包含1个独立序列的批次（批次大小为1），每个序列包含5个时间步（序列长度为5），并且每个时间步的输入向量大小为3（`input_size=3`）。**为了简化起见，批次大小为1而不是2.**

```py
import torch
import torch.nn as nn

torch.manual_seed(42)
rnn = nn.RNN(input_size=3, hidden_size=10,num_layers=1, batch_first=True)
input_data= torch.randn(1,5,3)
output, hidden = rnn(input_data)
print(input_data)
print(input_data.shape,output.shape,hidden.shape)
```

**提取权重和偏置**

首先，我们需要从PyTorch定义的RNN模型中提取权重和偏置参数。

```python
W_xh = rnn.weight_ih_l0.data
W_hh = rnn.weight_hh_l0.data
b_h = rnn.bias_ih_l0.data + rnn.bias_hh_l0.data
print(W_xh.shape, W_hh.shape, b_h.shape)# torch.Size([10, 3]) torch.Size([10, 10]) torch.Size([10])
```

<img src="./assets/image-20240317165528202.png" alt="image-20240317165528202" style="zoom:67%;" />



- `l0`表示**这些参数属于第一层（索引从0开始）的RNN层。**如果一个RNN有多个层（例如，`num_layers > 1`），那么对于每一层，都会有一组相应的参数，如`weight_ih_l1`和`weight_hh_l1`表示第二层的权重，以此类推。

- **$W_{xh}$：输入到隐藏状态的权重矩阵**
  - 形状：`[10, 3]`。这是因为每个输入向量的大小为3，而隐藏状态的大小（即隐藏层的特征数量）为10。$W_{xh}$负责将输入向量从3维映射到10维的隐藏状态空间。

2. **$W_{hh}$：隐藏状态到隐藏状态的权重矩阵**
   - 形状：`[10, 10]`。由于RNN的隐藏状态在每个时间步都是10维的，$W_{hh}$用于将前一时间步的隐藏状态（10维）映射到当前时间步的隐藏状态（同样是10维）。这个矩阵捕捉了序列中的时间依赖性。

3. **$b_h$：隐藏层的偏置项**
   - 形状：`[10]`。偏置项是加到激活函数之前的权重和的，用于增加模型的灵活性和偏移能力。由于隐藏状态的大小为10，所以偏置项也是10维的。

**手动实现RNN前向传播**
![image-20240318152301132](./assets/image-20240318152301132.png)

接着，我们将定义一个函数`manual_rnn_forward`来模拟RNN的前向传播过程。

根据公式:$$ h_t = f(W_{xh}x_t + W_{hh}h_{t-1} + b_h) $$

```python
def manual_rnn_forward(X, W_xh, W_hh, b_h):
    # 由于输入是单个批次，我们直接取第一个批次的数据
    seq_len, input_size = X.shape
    hidden_size = W_hh.shape[0]
    h_prev = torch.zeros(hidden_size)
    outputs = []

    for t in range(seq_len):
        x_t = X[t, :]  # 获取当前时间步的输入
        # 按照原始公式顺序进行计算
        h_t = torch.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
        outputs.append(h_t.unsqueeze(0))
        h_prev = h_t

    print("return:",len(outputs),outputs[0].shape,torch.cat(outputs, dim=0).shape, h_t.unsqueeze(0).shape)
    return torch.cat(outputs, dim=0), h_t.unsqueeze(0)

# 获取单个批次的输入数据
input_data = input_data[0]  # 从批次中取出单个序列

# 使用修改后的函数进行前向传播计算
manual_output, manual_hidden = manual_rnn_forward(input_data, W_xh, W_hh, b_h)
```

- **计算流程**：对于输入序列$x_1, x_2, x_3,\ldots$，在第一次循环迭代中，使用$x_1$和初始隐藏状态$h_0$（通常初始化为零）来计算$h_1$。在第二次迭代中，使用$x_2$和刚刚计算得到的$h_1$来计算$h_2$，以此类推。这种依赖前一时间步隐藏状态的方式使得RNN能够捕捉序列中的时间动态。

- **输出收集**：计算得到的每个时间步的隐藏状态$h_t$会被收集起来，形成输出序列。这些输出可以直接用作任务的预测结果，也可以进一步被用于计算最终的输出$y_t$。

- **`unsqueeze`**操作的作用是在指定位置增加一个维度。在这段代码中，`h_t.unsqueeze(1)`是将`h_t`从形状`[hidden_size]`转变为`[1, hidden_size]`。这样做的目的是为了使得每个时间步的输出可以沿着一个新的维度被拼接起来。

- **outputs**:是一个列表，其中的元素是`torch.Size([1, 10])`的张量。

- **`torch.cat(outputs, dim=0)`**的效果是将整个序列的隐藏状态合并成一个二维张量，其中第一个维度表示时间步，第二个维度表示每个时间步的隐藏状态的维度。

- **`return h_t.unsqueeze(0)`**:最后一个隐藏状态。

**调用手动实现的RNN并比较结果**

最后，我们将使用手动实现的RNN函数计算前向传播，然后与PyTorch的`nn.RNN`模块的输出进行比较。

```python
# 比较结果
print("Modified manual implementation (single batch):")
print("Output shape:", manual_output.shape)
print("Hidden state shape:", manual_hidden.shape)

# 验证输出与隐藏状态是否接近PyTorch的实现
output = output[0]  # 从PyTorch的输出中取出对应单个批次的输出
hidden = hidden[0]  # 对于单层RNN，取出对应单个批次的最后一个隐藏状态

print("Output close:", torch.allclose(manual_output, output, atol=1e-4))
print("Hidden state close:", torch.allclose(manual_hidden, hidden, atol=1e-4))
```

其中：
```python
torch.allclose(a, b, atol=tolerance)
```

- `a`：第一个张量。
- `b`：第二个张量，与`a`进行比较。
- `atol`（可选）：绝对容忍度。两个张量在每个元素上的差的绝对值必须小于或等于这个容忍度才被认为是接近的。

这段代码的执行将展示手动实现的RNN与PyTorch `nn.RNN`模块计算得到的输出和最后一个隐藏状态是否接近，以`atol=1e-4`为容忍度进行比较。如果结果显示两者非常接近，这表明我们的手动实现是正确的。









#### RNN拟合正弦波

在传统的监督学习任务中，如回归或分类，模型是**基于输入$x$来预测一个标签或输出**$f(x)$。然而，在处理序列数据，特别是进行序列生成或预测任务时，**模型的目标变成了基于序列的先前元素来预测序列的下一个元素。**

在这个正弦波预测的例子中，**模型接收的输入$y_{t-1}$实际上是序列中的一个元素，而模型的任务是预测该序列的下一个元素$y_t$。换句话说，给定序列的前$n$个值，模型需要预测第$n+1$个值。**这种类型的任务通常称为时间序列预测。

**生成数据集**

首先，我们生成正弦波数据集。

![image-20240319084624017](./assets/image-20240319084624017.png)

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 生成正弦波数据集
torch.manual_seed(44)  # 设置随机种子，保证每次运行结果一致
x = np.linspace(0, 100, 1000)
y = np.sin(x)
print("shape of x:",x.shape)
print("x",x[:20])
print("shape of y:",y.shape)
print("y",y[:20])

# 将数据转换为PyTorch张量
X = torch.tensor(y[:-1], dtype=torch.float32).view(-1, 1, 1)
Y = torch.tensor(y[1:], dtype=torch.float32).view(-1, 1, 1)
print("shape of X:",X.shape)
print("X",X[:20])
print("shape of Y:",Y.shape)
print("Y",Y[:20])

# 可视化部分数据集
plt.figure(figsize=(10,5))
plt.plot(x[:100], y[:100], label='Sin wave')
plt.title('Sin wave data')
plt.legend()
plt.show()
```

![image-20240314103515948](.\assets\image-20240314103515948.png)

![image-20240314095946491](.\assets\image-20240314095946491.png)

解释：

- `np.linspace`生成了**1000个介于0到100之间的均匀间隔的点**。
- `np.sin(x)`是**对数组`x`中的每一个元素应用正弦函数**，得到一个形状与`x`相同的数组。
- 在NumPy中，`(1000,)`表示的是一个含有1000个元素的一维数组。
- `X`是从`y`中**除了最后一个元素外所有元素**创建的PyTorch张量，并且我们通过`.view(-1, 1, 1)`调整了它的形状。`-1`在这里表示自动计算这个维度的大小，使得总的元素数保持不变。因此，`X`的形状应该是`(999, 1, 1)`，表示有999个时间步，每个时间步有1个特征，总共只有一个批次。
- `Y`与`X`类似，也是从`y`中创建的，但包含的是**从第二个元素到最后一个元素。**因此，`Y`的形状也是`(999, 1, 1)`，与`X`相同。
- `X`和`Y`的值被微微错开（即`X`是从`y`的第一个元素到倒数第二个元素，而`Y`是从`y`的第二个元素到最后一个元素），是为了**创建一个预测任务：给定当前的`sin`值（通过`X`表示），预测下一个时间点的`sin`值（通过`Y`表示）**。这种设置模仿了许多实际场景中的序列预测任务，例如，根据过去的天气数据预测未来的天气，或者根据过去的股价预测未来的股价。
- 对于时刻`t`,**`X[t]=sin(t)`而`Y[t]=sin(t+1)`。**
- `plot`填入x和y两组数据，绘制了`x`和`y`数组的前100个元素组成的曲线。
- `plt.legend()`：会在图表上显示一个**图例**，类似于下面这样。
  ![image-20240314141829480](.\assets\image-20240314141829480.png)

**定义RNN模型**

我们已经生成了正弦波数据集并可视化了其一部分。接下来，我们将定义一个简单的RNN模型来预测这个正弦波的未来值。

我们的RNN模型将非常简单，包括**一个RNN层，后面跟着一个线性层**来输出预测值。我们将使用PyTorch的`nn.RNN`类来创建这个RNN层，然后添加一个线性层进行输出值的转换。

让我们定义这个模型。

```py
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x


# 模型参数
input_size = 1
hidden_size = 100
output_size = 1

# 实例化模型
model = SimpleRNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

```

**`SimpleRNN`类继承自`nn.Module`类**，这是PyTorch中所有神经网络模块的基类。

通过调用`super().__init__()`，`SimpleRNN`类的构造器可以**调用其父类`nn.Module`的构造器**，这是初始化父类对象的标准方式。

![image-20240318232638103](./assets/image-20240318232638103.png)

- `input_size`：每个时间步的输入特征数量。在这个例子中，**每个时间步的输入是正弦波的一个值，因此`input_size=1`。**
- `hidden_size`：RNN隐藏层的特征数量。这决定了隐藏状态的维度，也就是RNN内部每个时间步计算得到的向量的大小。
  - **没有`batch_first=True`：此时RNN期望的输入是一个三维张量，其形状为`(seq_len, batch, input_size)`，恰好对应输入的张量形状`[999,1,1]`。因为只有一个曲线需要预测，批次数为1。在每一个时间点上，只有一个输入，也就是$x_t$`=sin(t)`，因此`input_size=1`。总共有999个时间点，因此`seq_len=999`。**
  - 之前说过：RNN期望的输入是一个三维张量，其形状为`(seq_len, batch, input_size)`，除非你将`batch_first=True`传递给RNN构造函数，在这种情况下输入形状应该是`(batch, seq_len, input_size)`。


**RNN层处理后返回两个值：输出【对应手写rnn当中的outputs】和最新的隐藏状态【对应手写rnn当中的$h_t$】。**在这里，我们只关心输出（也用`x`表示），而忽略了最新的隐藏状态（用`_`表示，这是一个常见的习惯用法，用于忽略不需要的返回值）。

**`nn.Linear`创建了一个线性层（全连接层），将RNN的输出映射到最终的预测值。**它的参数含义如下：

- `in_features`：**输入特征的数量，应与RNN隐藏层的输出大小相匹配**，即`hidden_size=10`。
- `out_features`：**输出特征的数量**。在这个例子中，**我们的目标是预测下一个正弦波的值，因此输出大小为1**，即`output_size=1`。
- 之前说过，**线性层的本质是：将一个向量(样本)映射到另一个向量。**
- 给定输入矩阵$X$，线性层的权重$W$和偏置$b$，**线性层的输出$Y$可以通过下面的公式计算：**

  $$ Y = XW^T + b $$

  这里：

  - **$X$ 是输入矩阵，假设其形状为$[N, \text{in\_features}]$，其中$N$是批量大小【样本数】，$\text{in\_features}$是每个输入样本的特征数量。**
  - **$W$ 是权重矩阵，其形状为$[\text{out\_features}, \text{in\_features}]$，其中$\text{out\_features}$是输出特征的数量。**
  - **$b$ 是偏置向量，形状为$[\text{out\_features}]$。**
  - $Y$ 是输出矩阵，形状为$[N, \text{out\_features}]$

我们已经定义了一个简单的RNN模型，它由一个RNN层和一个线性层组成。接下来，我们将训练这个模型来预测正弦波的未来值。

**训练模型**

为了训练模型，我们将使用均方误差（MSE）作为损失函数，使用Adam作为优化器。我们将通过一定数量的迭代（或称为"epoch"）来训练模型，每次迭代都会遍历整个数据集。

让我们开始训练过程。

```py
# 训练参数
learning_rate = 0.01 # 学习率
epochs = 150 # 训练轮数

# 损失函数和优化器
criterion = nn.MSELoss() # 均方误差损失，计算方法：(y_true - y_pred) ** 2 求和取平均
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam优化器

# 训练模型
losses = []  # 用于记录每个epoch的损失值
for epoch in range(epochs):
    model.train() # 确保模型处于训练模式，因为PyTorch中有一些层在训练和评估模式下行为不同
    optimizer.zero_grad()  # 清除之前的梯度
    output = model(X)  # 前向传播
    loss = criterion(output, Y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数，主要工作之一确实是梯度下降

    losses.append(loss.item()) # 记录损失值
    if epoch % 10 == 0: # 每10个epoch打印一次损失值
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 绘制损失下降图
plt.figure(figsize=(10, 5)) # 设置画布大小
plt.plot(losses, label='Training Loss') # 绘制损失值曲线
plt.title('Training Loss') # 设置图标题
plt.xlabel('Epoch') # 设置x轴标签
plt.ylabel('Loss') # 设置y轴标签
plt.legend() # 显示图例
plt.show() # 显示图像
```

训练过程已经完成，我们观察到随着训练的进行，损失值逐渐降低，这表明模型在学习如何预测正弦波的未来值。
![image-20240318173149611](./assets/image-20240318173149611.png)

**进行预测并绘制结果**

现在，我们将使用训练好的模型对测试集(不同区间上的波形)进行预测，并将预测结果与实际正弦波进行比较。

让我们进行预测并绘制实际结果与预测结果的对比图。

```py
# 生成测试数据集
x_test = np.linspace(100, 110, 100)# 生成100个点,从100到110之间
y_test = np.sin(x_test)# 生成对应的sin值

# 将测试数据转换为PyTorch张量
X_test = torch.tensor(y_test[:-1], dtype=torch.float32).view(-1, 1, 1)
# 从测试数据中取出前999个点，转换为PyTorch张量，形状为(999, 1, 1)

# 使用模型进行预测
model.eval()  # 确保模型处于评估模式
with torch.no_grad():
    predictions_test = model(X_test).view(-1).numpy()# 使用模型进行预测，得到的预测值的shape为(999, 1, 1)，需要将其转换为一维数组

# 绘制实际值和预测值的对比图
plt.figure(figsize=(10,5))# 设置画布大小
plt.plot(x_test[1:], y_test[1:], label='Actual Sin wave', color='blue')# 绘制实际值
plt.plot(x_test[1:], predictions_test, label='Predicted Sin wave', color='red', linestyle='--')# 绘制预测值
plt.title('Sin wave prediction on test data (x in [100, 110])')# 设置图标题
plt.xlabel('x')# 设置x轴标签
plt.ylabel('sin(x)')# 设置y轴标签
plt.legend()# 显示图例
plt.show()# 显示图像
```

![image-20240318173023531](./assets/image-20240318173023531.png)

在对比图中，我们可以看到蓝色曲线表示实际的正弦波，而红色虚线表示我们的RNN模型预测的正弦波。虽然存在一些差异，但整体上模型能够较好地捕捉到正弦波的模式并进行预测。

这表明我们的简单RNN模型已经学习到了如何根据历史数据来预测正弦波的未来值。通过调整模型参数、增加训练迭代次数或改进模型结构，可能会进一步提高预测的准确性。





### GRU

门控循环单元（Gated Recurrent Unit, GRU）是一种特殊的循环神经网络（RNN），它通过引入更新门（Update Gate）和重置门（Reset Gate）来解决标准RNN的梯度消失问题。GRU的计算比标准RNN复杂，但我们也可以通过类似的方法简化其计算。

GRU的更新过程可以分为三个主要步骤：更新门计算、重置门计算和候选隐藏状态计算。以下是这些步骤的详细公式：

1. **更新门（Update Gate）**:
   $$ z_t = \sigma(W_{xz} x_t + W_{hz} h_{t-1} + b_z) $$

2. **重置门（Reset Gate）**:
   $$ r_t = \sigma(W_{xr} x_t + W_{hr} h_{t-1} + b_r) $$

3. **候选隐藏状态（Candidate Hidden State）**:
   $$ \tilde{h}_t = \tanh(W_{xh} x_t + W_{hh} (r_t \odot h_{t-1}) + b_h) $$

4. **最终隐藏状态（Final Hidden State）**:
   $$ h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t $$

这里，$\sigma$代表sigmoid激活函数，$\odot$代表元素乘积（Hadamard product），$W$和$b$分别代表权重矩阵和偏置项。

#### GRU的简化公式

为了简化GRU的计算，我们可以通过合并权重矩阵和向量来减少重复的计算。具体地，我们可以将输入$x_t$和前一个隐藏状态$h_{t-1}$竖直拼接成一个新的大向量，同时将每一对权重矩阵水平拼接成新的大权重矩阵。对于更新门和重置门的计算：

1. 合并的更新门和重置门权重矩阵：
   $$ W_{z} = [W_{xz}; W_{hz}], \quad W_{r} = [W_{xr}; W_{hr}] $$

2. 合并的权重矩阵用于候选隐藏状态：
   $$ W_{h} = [W_{xh}; W_{hh}] $$

然后，将输入和前一个隐藏状态拼接：
   $$ [x_t; h_{t-1}] $$

这样，我们可以同时计算更新门和重置门，以及候选隐藏状态的一部分。但是，由于GRU中涉及的元素乘积（$\odot$），我们不能完全以单一矩阵乘法形式简化整个计算过程。候选隐藏状态的计算仍然需要单独处理重置门与前一个隐藏状态的元素乘积。

因此，尽管我们可以通过拼接来简化一部分权重矩阵的乘法，GRU的计算过程中仍有部分无法通过这种方式完全简化，特别是因为重置门作用于前一个隐藏状态的机制。



#### 配置环境

```py
pip install xmnlp
```

使用`conda install xmnlp`可能会报错



![image-20240318084918171](./assets/image-20240318084918171.png)







#### 性能指标

![image-20240321113008094](./assets/image-20240321113008094.png)

**问题：**

- 习惯上，我们把标签为1叫做正（Positive），标签0叫做负（Negative）。

- 假设有20个样本，预测标签为1且实际标签为1的样本（TP，True Positives）有8个，预测标签为1且实际标签为0的样本（FP，False Positives）有2个，预测标签为0且实际标签为1的样本（FN，False Negatives）有3个，预测标签为0且实际标签为0（TN，True Negatives）的样本有7个。【**命名原理：预测的标签决定是Positives（1）还是Negatives（0），用预测和实际是否相符决定叫真还是假**】
- 计算出标签1上的精确率，召回率，F1，然后计算出标签0上的精确率，召回率，F1，求出最终的准确率。

**答：**

```py
# 定义给定的值
TP = 8  # 预测标签为1且实际标签为1
FP = 2  # 预测标签为1且实际标签为0
FN = 3  # 预测标签为0且实际标签为1
TN = 7  # 预测标签为0且实际标签为0

# 计算标签1上的精确率(Precision)，召回率(Recall)和F1分数
precision_1 = TP / (TP + FP)
recall_1 = TP / (TP + FN)
F1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)

# 计算标签0上的精确率，召回率和F1分数
precision_0 = TN / (TN + FN)
recall_0 = TN / (TN + FP)
F1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)

# 计算准确率
accuracy = (TP + TN) / (TP + TN + FP + FN)

print(
{
    "precision_1": precision_1,
    "recall_1": recall_1,
    "F1_1": F1_1,
    "precision_0": precision_0,
    "recall_0": recall_0,
    "F1_0": F1_0,
    "accuracy": accuracy
}
)
```

基于给定的样本信息，我们得到以下结果：

对于标签1（正类）：
- 精确率（Precision）为0.8，
- 召回率（Recall）为0.727，
- F1分数为0.762。

对于标签0（负类）：
- 精确率为0.7，
- 召回率为0.778，
- F1分数为0.737。

整体的准确率（Accuracy）为0.75。

**解释**：

上述公式用于计算机器学习模型在二分类问题中的性能指标，包括精确率（Precision），召回率（Recall），F1分数（F1 Score），以及准确率（Accuracy）。这些指标基于四个基本概念：真正类（TP），假正类（FP），假负类（FN），真负类（TN）。

**真正类（True Positives, TP）**
指的是模型正确预测为正类的样本数量。

**假正类（False Positives, FP）**
指的是模型错误预测为正类的样本数量，但实际上它们属于负类。

**假负类（False Negatives, FN）**
指的是模型错误预测为负类的样本数量，但实际上它们属于正类。

**真负类（True Negatives, TN）**
指的是模型正确预测为负类的样本数量。

基于这些定义，我们可以解释上述的7个公式：

**标签1上的精确率（Precision）**
$$ \text{Precision} = \frac{TP}{TP + FP} $$
**表示在所有预测为正类的样本中，实际为正类的比例（对了多少）。精确率高表示模型在预测为正类的时候，确实性较高。**

**标签1上的召回率（Recall）**
$$ \text{Recall} = \frac{TP}{TP + FN} $$
**表示在所有实际为正类的样本中，被模型正确预测出来的比例（对了多少）。召回率高表示模型能够很好地捕捉到正类样本。**

**标签1上的F1分数**
$$ \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$
**F1分数是精确率和召回率的调和平均值，用于平衡精确率和召回率。当模型在精确率和召回率之间存在差异时，F1分数是一个有用的度量。**

标签0上的精确率、召回率和F1分数
对于负类（标签0），这些指标的计算方式与正类（标签1）相同，但是基于TN和FN的计算，分别关注于负类的预测性能。

**准确率（Accuracy）**
$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$
**准确率表示模型正确预测（无论正类或负类）的样本比例，是最直观的性能衡量指标。**

































































