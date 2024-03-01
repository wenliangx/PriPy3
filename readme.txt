类：BasePlane
作用：包含最基本的位置、速度、速率限制和加速度限制

类：BaseCalculateFunc
作用：用于计算飞机最佳数值的方法接口

类：Value
作用：用于计算飞机行动最佳数值，调用BaseCalculateFunc并默认为pso方法

类：Plane
作用：是BasePlane的子类，相比起BasePlane，多添加了Value类，可在后期用于实现分布式架构

类：Priority
作用：包含两队Plane，主要为了计算优势矩阵matrix
