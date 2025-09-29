import math
import numpy as np

class VRPTWInstance:
    """VRPTW问题实例类：读取数据并初始化问题参数"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.vehicle_info = {}  # 车辆信息（编号、容量）
        self.warehouse = {}  # 仓库信息
        self.customers = []  # 所有客户（含仓库）
        self.ordinary_customers = []  # 普通客户（不含仓库）
        self.distance_matrix = None  # 距离矩阵

        self._read_data()
        self._calculate_distance_matrix()

    def _read_data(self):
        """解析数据文件"""
        try:
            with open(self.file_path, 'r') as f:
                data = f.read()
        except FileNotFoundError:
            print(f"错误：文件 {self.file_path} 未找到")
            return

        # 分割车辆和客户数据段
        try:
            vehicle_section, customer_section = data.split('CUSTOMER', 1)
        except ValueError:
            print("错误：数据格式异常，未找到'CUSTOMER'分隔符")
            return

        # 解析车辆信息（第一行有效数据）
        vehicle_lines = [line.strip() for line in vehicle_section.split('\n') if line.strip()]
        for line in vehicle_lines:
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                self.vehicle_info = {
                    'number': int(parts[0]),
                    'capacity': int(parts[1])
                }
                break
        if not self.vehicle_info:
            print("错误：未找到有效车辆信息")
            return

        # 解析客户信息（含仓库）
        customer_lines = [line for line in customer_section.split('\n') if line.strip()]
        if not customer_lines:
            print("错误：客户数据为空")
            return

        # 提取客户字段（跳过标题行）
        for line in customer_lines[1:]:
            parts = [p for p in line.split() if p]
            if len(parts) >= 7:
                customer = {
                    'cust_no': int(parts[0]),  # 客户编号（0为仓库）
                    'xcoord': int(parts[1]),  # X坐标
                    'ycoord': int(parts[2]),  # Y坐标
                    'demand': int(parts[3]),  # 需求量
                    'ready_time': int(parts[4]),  # 最早可用时间
                    'due_date': int(parts[5]),  # 最晚接受时间
                    'service_time': int(parts[6])  # 服务时间
                }
                self.customers.append(customer)

        # 分离仓库和普通客户
        try:
            self.warehouse = next(cust for cust in self.customers if cust['cust_no'] == 0)
            self.ordinary_customers = [cust for cust in self.customers if cust['cust_no'] != 0]
        except StopIteration:
            print("错误：未找到仓库（cust_no=0）")
            return

    def _calculate_distance_matrix(self):
        """计算欧氏距离矩阵（仓库+普通客户）"""
        if not self.warehouse or not self.ordinary_customers:
            return

        # 构建节点列表（仓库在前，普通客户在后）
        nodes = [self.warehouse] + self.ordinary_customers
        n = len(nodes)
        self.distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = nodes[i]['xcoord'] - nodes[j]['xcoord']
                    dy = nodes[i]['ycoord'] - nodes[j]['ycoord']
                    self.distance_matrix[i][j] = np.hypot(dx, dy)


if __name__ == "__main__":
    print("Testing VRPTWInstance with invalid file path")
    from vrptw_instance import VRPTWInstance
    inst = VRPTWInstance("D:/vrptw/homberger_200_customer_instances/C1_2_1.TXT")
    if inst.vehicle_info:
        print("Vehicle info loaded:", inst.vehicle_info,inst.customers)
    else:
        print("Vehicle info not loaded (as expected)")
