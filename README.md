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
![image-20240206103758915](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206103758915.png)

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

![image-20240206000643472](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206000643472.png)

1. **逐元素乘法（哈达玛积）**：
   两个矩阵的逐元素乘法。

   ```python
   a = torch.tensor([[1, 2], [3, 4]])
   b = torch.tensor([[5, 6], [7, 8]])
   c = a * b  # 逐元素相乘
   print(c)
   ```

2. **矩阵除法**：
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

   ![image-20240206100829147](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206100829147.png)

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

![image-20240206101057868](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206101057868.png)

1. **逆矩阵**：


 计算可逆矩阵的逆。

   ```python
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
a_inverse = torch.inverse(a)
print(a_inverse)
#tensor([[-2.0000,  1.0000],
#        [ 1.5000, -0.5000]])
   ```

![image-20240206101901034](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206101901034.png)
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
![image-20240206102905942](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206102905942.png)

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
![image-20240206105143205](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206105143205.png)

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
![image-20240206114444888](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206114444888.png)

#### PyTorch 计算的特征向量

在 PyTorch 中，求得的特征向量是标准化的，这意味着它们的长度（或范数）被归一化为 1。在你的例子中，特征向量是复数，且其模长（绝对值）被标准化为 1。

以第一个特征向量为例：`[0.7071+0.0000j, 0.0000-0.7071j]`。这个向量的模长为：

$$
 \sqrt{(0.7071)^2 + (-0.7071)^2} = \sqrt{0.5 + 0.5} = \sqrt{1} = 1 
$$


所以，PyTorch 返回的特征向量是单位特征向量。它选择 $ 0.7071 $（即 $ \frac{1}{\sqrt{2}} $）是因为这样可以使特征向量的模长为 1，满足标准化的要求。这是数值计算中常用的做法，以保证结果的一致性和可比较性。
![image-20240206114106999](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206114106999.png)

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
![image-20240206120334406](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206120334406.png)

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

![image-20240206121133197](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206121133197.png)

#### 激活函数和其他非线性函数

![sigmoid_functions](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\sigmoid_functions.jpg)

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

![image-20240206163050782](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206163050782.png)

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

![image-20240206163116364](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206163116364.png)

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

![image-20240206163212678](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206163212678.png)

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

![image-20240206164324072](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206164324072.png)

```python
a = torch.tensor([[-100, 0], [1,100]])
softmax = torch.softmax(a.float(), dim=0)
print(softmax)
'''tensor([[1.4013e-44, 3.7835e-44],
        [1.0000e+00, 1.0000e+00]])'''
```

当你在 `dim=0` （按列）应用 softmax 时，每列的元素被转换为概率，使得每列的概率总和为 1。
![image-20240206164521479](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206164521479.png)

#### 5. Leaky ReLU

Leaky ReLU 是 ReLU 的一个变体，允许负输入有一个小的正斜率，定义为 $\text{LeakyReLU}(x) = \max(0.01x, x) $。

```python
a = torch.tensor([[1, -2], [3,-4]])
leaky_relu = torch.nn.functional.leaky_relu(a.float(), negative_slope=0.01)
print(leaky_relu)
'''tensor([[ 1.0000, -0.0200],
        [ 3.0000, -0.0400]])'''
```

![image-20240206165432886](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206165432886.png)

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

![image-20240206170111803](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206170111803.png)
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
![image-20240206180803978](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206180803978.png)

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
![image-20240206180912973](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206180912973.png)

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

![image-20240206193748618](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206193748618.png)

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

![image-20240206200632696](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206200632696.png)
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
![image-20240206202743530](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240206202743530.png)

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

#### 线性层nn.Linear

```python
import torch
import torch.nn as nn

# 定义输入数据
input_data = torch.randn(2, 4)  # 2个样本，4个特征
print('Input Data:', input_data)
'''tensor([[-0.0651,  0.1765, -0.5925, -0.6215],
        [-0.4160,  0.4052,  0.6214,  0.0385]])'''

# nn.Linear: 全连接层/线性层
linear_layer = nn.Linear(4, 3)  # 输入特征维度为4，输出特征维度为3
print('Linear Layer:',linear_layer.weight.data, linear_layer.bias.data)
'''tensor([[-0.3712,  0.1380,  0.1242, -0.1603],
        [-0.0283,  0.3212, -0.4816,  0.2166],
        [-0.1501, -0.2759,  0.1762, -0.0754]])'''
'''tensor([-0.2100, -0.4055, -0.2082])'''

output = linear_layer(input_data)
print("Linear Layer Output:")
print(output)
print(input_data@linear_layer.weight.t()+linear_layer.bias)
print()
'''tensor([[-0.1355, -0.1963, -0.3046],
        [ 0.0712, -0.5546, -0.1509]], grad_fn=<AddmmBackward0>)'''
'''tensor([[-0.1355, -0.1963, -0.3046],
        [ 0.0712, -0.5546, -0.1509]], grad_fn=<AddBackward0>)'''
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
![image-20240208133706466](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240208133706466.png)

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

![image-20240208143402662](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240208143402662.png)

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
![image-20240208175803651](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240208175803651.png)

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

  ![image-20240208180507220](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240208180507220.png)

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


![image-20240208182052802](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240208182052802.png)![image-20240208181629799](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240208181629799.png)

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

2. **i,j,k,l是什么**
   
   - `i`：这个变量代表的是输入图像的批次索引。在您的例子中，`input_image` 包含两张图像，所以 **`i` 会从 0 到 1 变化，分别对应批次中的第一张和第二张图像**。
   
   - `j`：这个变量代表的是输出通道的索引，或者说是卷积核的索引。在您的例子中，有两个卷积核（因为 `out_channels=2`），**所以 `j` 会从 0 到 1 变化，分别对应两个不同的卷积核**。
   
   - `k` 和 `l`：这两个变量代表的是在输出特征图上的空间位置**。`k` 代表输出特征图的行索引，而 `l` 代表列索引。**它们决定了卷积核在输入图像上的位置，这取决于步长和卷积核的大小。
   
   因此，循环中的这些索引结合起来，表示了在输入图像的批次 (`i`) 上，使用特定的卷积核 (`j`)，在特定的位置 (`k`, `l`) 进行卷积操作，从而生成输出特征图的每个元素。

![image-20240208193111241](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240208193111241.png)

1. **核心卷积代码解释**：

   在手动实现的卷积操作中，最关键的部分是这一行代码：

   ```python
   output[i, j, k, l] = torch.sum(input_padded[i, :, h_start:h_end, w_start:w_end] * weights[j]) + bias[j]
   ```

   这里的每个元素代表了：

   - `output[i, j, k, l]`：这**表示输出张量中的一个特定元素**。其中：
     
     - `i` 代表当前处理的是**输入批次中的第几张图像，同时也是输出的第i张图像(一一对应)**。
     - **`j` 代表当前使用的是第几个卷积核，因此也对应于输出特征图的第几个通道。**
     - **`k` 和 `l` 分别代表输出特征图中的行和列索引。**
     
   - `input_padded[i, :, h_start:h_end, w_start:w_end]`：这部分是对输入图像的选取。其中：
     
     - `i` 同上，要输出第i张图像，就要用输入的第i张图像进行卷积。
     - `:` 表示选择该图像的所有通道,也就是小立方体深度应该是满的。
     - 要输出第k行，第l列的数值，**卷积核应该在h_start:h_end这两行之间，在w_start:w_end这两列之间。**
     
     **这样我们就从若干张输入图像当中挑出来第i张，看作一个大立方体，并且从其中抠出来了一个小立方体，和第j个卷积核进行卷积。**
     
   - `weights[j]`：这是第 `j` 个卷积核的权重。

   - `torch.sum( ... )`：这个函数计算了输入图像的选定区域和卷积核权重之间的元素乘积的总和。

   - `+ bias[j]`：在求和后，加上对应卷积核的偏置值,**一个卷积核有一个偏置,用了第j个卷积核应该加上偏置bias[j]**。

   因此，整个表达式计算了输入图像的一个局部区域与一个卷积核的卷积，并生成了输出特征图的一个特定位置的值。这个过程在所有输入图像、所有输出通道、以及输出特征图的每个位置上重复进行，从而生成完整的输出特征图。
   ![image-20240208193442876](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240208193442876.png)

不难验证用系统库的卷积**nn.Conv2d**与我们手写的卷积效果完全相同，也就是正确的。



### pytorch构建神经网络：线性回归

![image-20240207175356280](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240207175356280.png)

线性回归是一种预测连续值的监督学习算法，多变量线性回归意味着模型的输入包含多个特征（变量）。以下是一步一步创建和训练一个简单的多变量线性回归模型的过程：

#### 步骤 1: 准备数据

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
![image-20240208120524170](C:\Users\86157\AppData\Roaming\Typora\typora-user-images\image-20240208120524170.png)

#### 步骤 2: 定义模型

我们将使用 PyTorch 的 `nn.Module` 类来定义我们的模型。对于线性回归，我们可以使用 PyTorch 提供的 `nn.Linear` 层。
在 PyTorch 中，`LinearRegressionModel` 是您自定义的类的名字，并不是系统库的。您可以根据您的需求给这个类命名，但最好选择一个描述性的名称，以便清楚地表明类的功能。
当您定义一个类并在括号里写 `nn.Module` 时，您的类 `LinearRegressionModel` 成为 `nn.Module` 的子类。这意味着您的类继承了 `nn.Module` 的所有功能，包括参数管理、钩子函数、训练/评估模式切换等。

```python
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        # 定义模型的层
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        # 前向传播函数
        return self.linear(x)

# 实例化模型
model = LinearRegressionModel(input_size=2)
```

当您在模型上调用 `model(x)` 时，PyTorch 会自动调用 `forward` 方法。这是实现自定义模块时必须遵循的 PyTorch 框架的约定。
`LinearRegressionModel` 类继承自 `nn.Module`，并且重写了 `forward()` 方法。这意味着在创建 `LinearRegressionModel` 对象时，可以调用 `forward()` 方法来执行模型的前向传播，从而计算模型的输出。

#### 步骤 3: 定义损失函数和优化器

接下来，我们需要定义一个损失函数和一个优化器，用于训练模型。

```python
# 均方误差损失函数
loss_function = nn.MSELoss()

# 随机梯度下降优化器
optimizer = SGD(model.parameters(), lr=0.01)
```

#### 步骤 4: 训练模型

现在我们可以开始训练我们的模型了。

```python
# 训练模型
epochs = 1000  # 训练轮数
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清空过往梯度

    y_pred = model(x_data)  # 进行预测
    loss = loss_function(y_pred, y_data.unsqueeze(1))  # 计算损失

    loss.backward()  # 反向传播，计算当前梯度
    optimizer.step()  # 根据梯度更新网络参数

    # 每隔一段时间输出训练信息
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```

#### 步骤 5: 评估模型

训练完成后，我们可以查看模型的参数，看看它们是否接近真实的权重和偏置。

```python
print("模型参数:", model.linear.weight.data, model.linear.bias.data)
```

#### 完整代码

以上所有代码片段合并起来，就构成了一个完整的多变量线性回归模型训练过程。

这个例子展示了如何使用 PyTorch 构建和训练一个简单的线性回归模型。通过这个过程，您可以了解如何定义自己的模型、如何通过损失函数和优化器来迭代训练模型，以及如何评估模型的性能。
#### 步骤6：画图

对于两个特征变量的线性回归，您可以在三维空间中绘制一个平面来表示预测的模型，以及散点图来表示数据点。这可以使用 `matplotlib` 库中的 `mplot3d` 模块来完成。以下是示例代码：

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设模型已经训练完毕，并且我们有 model.linear.weight 和 model.linear.bias

# 创建一个新的图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据点
ax.scatter(x_data[:, 0].numpy(), x_data[:, 1].numpy(), y_data.numpy())

# 为了绘制平面，我们需要创建一个网格并计算相应的y值
x1_grid, x2_grid = torch.meshgrid(torch.linspace(-3, 3, 10), torch.linspace(-3, 3, 10))
y_grid = model.linear.weight[0, 0].item() * x1_grid + model.linear.weight[0, 1].item() * x2_grid + model.linear.bias.item()

# 绘制预测平面
ax.plot_surface(x1_grid.numpy(), x2_grid.numpy(), y_grid.numpy(), alpha=0.5)

# 设置坐标轴标签
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

# 显示图形
plt.show()
```

在这段代码中，我们首先绘制了数据点，然后创建了一个网格来代表特征空间，并使用模型的权重和偏置来计算网格上每个点的预测值，从而绘制了预测平面。

请注意，这段代码假设 `model` 已经被训练并且包含了线性回归模型的权重和偏置。此外，`x_data` 和 `y_data` 是原始数据集的特征和目标值。您可以调整代码中的数值和变量以匹配您的实际数据和模型。
