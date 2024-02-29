版本：2.0.0
类：plane priority

plane：
	属性：
		飞机的位置（直角坐标系）速度（球坐标系）
	 	更新的位置和速度
	方法：
		calculate_update_velocity：根据所给值和方法计算更新速度 
		calculate_update_position：根据更新速度计算更新位置
		translate_velocity：返回直角坐标系下的速度
		update：将当前的速度和位置更新为更新的位置和速度

priority：
	属性：
		两架飞机，优势矩阵和粒子群算法计算最佳策略值的相关参数
	方法：
		priority_d， priority_v， priority_a：分别计算距离，速度和角度优势
		single_priority：计算某一种策略下的飞机对飞机的优势	
		matrix_priority：粒子群算法计算最佳值，并通过调用single_priority计算各策略下的优势返回优势矩阵，本方法在初始化时自动调用
		