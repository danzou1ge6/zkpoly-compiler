use halo2curves::bn256;
use std::mem::size_of;
use std::ptr::copy_nonoverlapping;
use zkpoly_common::devices::DeviceType;
use zkpoly_cuda_api::stream::CudaStream;
use zkpoly_memory_pool::CpuMemoryPool;
use zkpoly_runtime::runtime::transfer::Transfer;
use zkpoly_runtime::scalar::ScalarArray;

// 定义测试用的Field类型
type TestField = bn256::Fr;

// 辅助函数：创建一个CPU上的ScalarArray，使用PinnedMemoryPool
fn create_cpu_array<const N: usize>(
    values: [TestField; N],
) -> (ScalarArray<TestField>, CpuMemoryPool) {
    let mut pool = CpuMemoryPool::new(10, size_of::<TestField>()); // 足够大的池
    let ptr = pool.allocate(N);
    unsafe {
        copy_nonoverlapping(values.as_ptr(), ptr, N);
    }
    let array = ScalarArray::new(N, ptr, DeviceType::CPU);
    (array, pool)
}

// 辅助函数：创建一个GPU上的ScalarArray
fn create_gpu_array<const N: usize>(
    values: [TestField; N],
    stream: &CudaStream,
) -> ScalarArray<TestField> {
    let (cpu_array, _pool) = create_cpu_array(values);
    let ptr = stream.allocate(N);
    let mut gpu_array = ScalarArray::new(
        N,
        ptr,
        DeviceType::GPU {
            device_id: stream.get_device(),
        },
    );
    cpu_array.cpu2gpu(&mut gpu_array, stream);
    gpu_array
}

#[test]
fn test_rotate_slice_basic() {
    let values = [
        TestField::from(0),
        TestField::from(1),
        TestField::from(2),
        TestField::from(3),
        TestField::from(4),
        TestField::from(5),
        TestField::from(6),
        TestField::from(7),
    ];
    let (mut array, _pool) = create_cpu_array(values);

    // 旋转2位，数组变为[6,7,0,1,2,3,4,5]
    array.rotate(2);

    // 测试普通切片（不跨边界）
    let slice = array.slice(3, 6);
    assert_eq!(slice.len, 3);
    // 检查切片内容是否正确（应该是[1,2,3]）
    for i in 0..3 {
        assert_eq!(slice[i], values[(i + 1) % 8]);
    }
}

#[test]
fn test_rotate_slice_cross_boundary() {
    let values = [
        TestField::from(0),
        TestField::from(1),
        TestField::from(2),
        TestField::from(3),
        TestField::from(4),
        TestField::from(5),
        TestField::from(6),
        TestField::from(7),
    ];
    let (mut array, _pool) = create_cpu_array(values);

    // 旋转1位，数组逻辑上变为[7,0,1,2,3,4,5,6]
    array.rotate(1);

    // 测试跨边界的切片（取[7, 0, 1]，这是不连续的内存区域）
    let slice = array.slice(0, 3);
    assert_eq!(slice.len, 3);
    // 验证切片内容
    assert_eq!(slice[0], values[7]);
    assert_eq!(slice[1], values[0]);
    assert_eq!(slice[2], values[1]);
}

#[test]
fn test_rotate_slice_to_array_transfer() {
    let stream = CudaStream::new(0);
    let values = [
        TestField::from(0),
        TestField::from(1),
        TestField::from(2),
        TestField::from(3),
        TestField::from(4),
        TestField::from(5),
        TestField::from(6),
        TestField::from(7),
    ];

    // 创建GPU数组并旋转
    let mut gpu_array = create_gpu_array(values, &stream);
    gpu_array.rotate(3); // 旋转3位，逻辑上变为[5,6,7,0,1,2,3,4]

    // 创建一个切片（包含跨边界区域）
    let slice = gpu_array.slice(1, 4); // 应该包含[6,7,0]

    // 创建目标数组并传输
    let ptr = stream.allocate(3);
    let mut dst_array = ScalarArray::new(
        3,
        ptr,
        DeviceType::GPU {
            device_id: stream.get_device(),
        },
    );

    // 从旋转后的切片传输到普通数组
    <ScalarArray<TestField> as Transfer>::gpu2gpu(&slice, &mut dst_array, &stream);

    // 验证传输结果
    let mut verify_pool = CpuMemoryPool::new(10, size_of::<TestField>());
    let mut verify_array = ScalarArray::new(3, verify_pool.allocate(3), DeviceType::CPU);
    <ScalarArray<TestField> as Transfer>::gpu2cpu(&dst_array, &mut verify_array, &stream);
    stream.sync();

    // 验证内容
    assert_eq!(verify_array[0], values[6]); // 6
    assert_eq!(verify_array[1], values[7]); // 7
    assert_eq!(verify_array[2], values[0]); // 0

    // 清理GPU内存
    stream.free(gpu_array.values);
    stream.free(dst_array.values);
}

#[test]
fn test_rotate_slice_to_rotate_array_transfer() {
    let stream = CudaStream::new(0);
    let values = [
        TestField::from(0),
        TestField::from(1),
        TestField::from(2),
        TestField::from(3),
        TestField::from(4),
        TestField::from(5),
        TestField::from(6),
        TestField::from(7),
    ];

    // 创建GPU数组并旋转
    let mut gpu_array = create_gpu_array(values, &stream);
    gpu_array.rotate(3); // 旋转3位，逻辑上变为[5,6,7,0,1,2,3,4]

    // 创建一个切片（包含跨边界区域）
    let slice = gpu_array.slice(1, 4); // 应该包含[6,7,0]

    // 创建目标数组并传输
    let ptr = stream.allocate(3);
    let mut dst_array = ScalarArray::new(
        3,
        ptr,
        DeviceType::GPU {
            device_id: stream.get_device(),
        },
    );
    dst_array.rotate(2);

    // 从旋转后的切片传输到普通数组
    <ScalarArray<TestField> as Transfer>::gpu2gpu(&slice, &mut dst_array, &stream);

    // 验证传输结果
    let mut verify_pool = CpuMemoryPool::new(10, size_of::<TestField>());
    let mut verify_array = ScalarArray::new(3, verify_pool.allocate(3), DeviceType::CPU);
    <ScalarArray<TestField> as Transfer>::gpu2cpu(&dst_array, &mut verify_array, &stream);
    stream.sync();

    // 验证内容
    assert_eq!(verify_array[0], values[6]); // 6
    assert_eq!(verify_array[1], values[7]); // 7
    assert_eq!(verify_array[2], values[0]); // 0

    // 清理GPU内存
    stream.free(gpu_array.values);
    stream.free(dst_array.values);
}

#[test]
fn test_rotate_array_to_rotate_slice_transfer() {
    let stream = CudaStream::new(0);
    let values = [
        TestField::from(0),
        TestField::from(1),
        TestField::from(2),
        TestField::from(3),
        TestField::from(4),
        TestField::from(5),
        TestField::from(6),
        TestField::from(7),
    ];

    // 创建旋转后的GPU目标数组
    let mut gpu_array = create_gpu_array(values, &stream);
    gpu_array.rotate(2); // 旋转2位，逻辑上变为[6,7,0,1,2,3,4,5]

    // 创建源数组
    let src_values = [
        TestField::from(10),
        TestField::from(11),
        TestField::from(12),
    ];
    let mut src_array = create_gpu_array(src_values, &stream);
    src_array.rotate(2); // 旋转2位，逻辑上变为[11,12,10]

    // 获取旋转数组的切片作为目标
    let mut dst_slice = gpu_array.slice(1, 4); // 跨边界区域[7,0,1]

    // 从普通数组传输到旋转后的切片
    <ScalarArray<TestField> as Transfer>::gpu2gpu(&src_array, &mut dst_slice, &stream);

    // 验证传输结果
    let mut verify_pool = CpuMemoryPool::new(10, size_of::<TestField>());
    let mut verify_array = ScalarArray::new(8, verify_pool.allocate(8), DeviceType::CPU);
    <ScalarArray<TestField> as Transfer>::gpu2cpu(&gpu_array, &mut verify_array, &stream);
    stream.sync();

    // 验证内容（检查整个数组，确保只有切片部分被修改）
    assert_eq!(verify_array[0], values[6]);
    assert_eq!(verify_array[1], src_values[1]); // 修改的部分
    assert_eq!(verify_array[2], src_values[2]); // 修改的部分
    assert_eq!(verify_array[3], src_values[0]); // 修改的部分
    assert_eq!(verify_array[4], values[2]); // 未修改的部分
    assert_eq!(verify_array[5], values[3]);
    assert_eq!(verify_array[6], values[4]);
    assert_eq!(verify_array[7], values[5]);

    // 清理GPU内存
    stream.free(gpu_array.values);
    stream.free(src_array.values);
}

#[test]
fn test_rotate_slice_to_slice_transfer() {
    let stream = CudaStream::new(0);
    let values = [
        TestField::from(0),
        TestField::from(1),
        TestField::from(2),
        TestField::from(3),
        TestField::from(4),
        TestField::from(5),
        TestField::from(6),
        TestField::from(7),
    ];

    // 创建两个旋转后的GPU数组
    let mut src_array = create_gpu_array(values, &stream);
    let mut dst_array = create_gpu_array(values, &stream);

    src_array.rotate(2); // 源数组旋转2位 [6,7,0,1,2,3,4,5]
    dst_array.rotate(3); // 目标数组旋转3位 [5,6,7,0,1,2,3,4]

    // 创建源切片和目标切片（都是跨边界的）
    let src_slice = src_array.slice(1, 4); // [7,0,1]
    let mut dst_slice = dst_array.slice(1, 4); // [6,7,0]

    // 从一个旋转切片传输到另一个旋转切片
    <ScalarArray<TestField> as Transfer>::gpu2gpu(&src_slice, &mut dst_slice, &stream);

    // 验证传输结果
    let mut verify_pool = CpuMemoryPool::new(10, size_of::<TestField>());
    let mut verify_array = ScalarArray::new(8, verify_pool.allocate(8), DeviceType::CPU);
    <ScalarArray<TestField> as Transfer>::gpu2cpu(&dst_array, &mut verify_array, &stream);
    stream.sync();

    // 验证内容
    assert_eq!(verify_array[0], values[5]);
    assert_eq!(verify_array[1], values[7]); // 第一个元素被修改为7
    assert_eq!(verify_array[2], values[0]); // 第二个元素被修改为0
    assert_eq!(verify_array[3], values[1]); // 第三个元素被修改为1
    assert_eq!(verify_array[4], values[1]); // 未修改的部分保持不变

    // 清理GPU内存
    stream.free(src_array.values);
    stream.free(dst_array.values);
}
