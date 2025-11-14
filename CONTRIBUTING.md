# Contributing to MonitorAI

Cảm ơn bạn đã quan tâm đến việc đóng góp cho MonitorAI! 🎉

## Code of Conduct

Hãy tôn trọng và lịch sự trong mọi tương tác.

## Cách đóng góp

### Báo cáo lỗi (Bug Reports)

1. Kiểm tra xem lỗi đã được báo cáo chưa trong [Issues](https://github.com/yourusername/MonitorAI/issues)
2. Nếu chưa, tạo issue mới với:
   - Mô tả rõ ràng về lỗi
   - Các bước để tái hiện lỗi
   - Môi trường (OS, Python version, Docker version)
   - Logs và error messages (nếu có)

### Đề xuất tính năng (Feature Requests)

1. Kiểm tra xem tính năng đã được đề xuất chưa
2. Tạo issue với label `enhancement`
3. Mô tả chi tiết về tính năng và use case

### Pull Requests

1. **Fork repository** và clone về máy local
2. **Tạo branch mới** từ `master`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Thực hiện thay đổi**:
   - Tuân theo coding style hiện có
   - Thêm comments cho code phức tạp
   - Cập nhật documentation nếu cần
   - Test thay đổi của bạn
4. **Commit changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
   - Sử dụng [Conventional Commits](https://www.conventionalcommits.org/):
     - `feat:` - Tính năng mới
     - `fix:` - Sửa lỗi
     - `docs:` - Thay đổi documentation
     - `style:` - Formatting, không ảnh hưởng logic
     - `refactor:` - Refactor code
     - `test:` - Thêm/sửa tests
     - `chore:` - Các thay đổi khác (build, config, etc.)
5. **Push và tạo Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Setup

### Yêu cầu

- Docker Desktop
- Conda environment `Grafotel` với Python 3.11+
- NVIDIA GPU (optional, cho GPU monitoring)

### Setup

1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/MonitorAI.git
   cd MonitorAI
   ```

2. Kích hoạt conda environment:
   ```powershell
   conda activate Grafotel
   ```

3. Cài đặt dependencies:
   ```powershell
   pip install -r llm-monitor/requirements.txt
   pip install -r gpu-exporter/requirements.txt
   ```

4. Start services:
   ```powershell
   .\start-all.ps1
   ```

## Coding Standards

### Python

- Sử dụng PEP 8 style guide
- Maximum line length: 120 characters
- Sử dụng type hints khi có thể
- Docstrings cho functions và classes
- Format code với `black` (nếu có)

### PowerShell

- Sử dụng 4 spaces cho indentation
- Comment rõ ràng cho các function phức tạp
- Sử dụng `Write-Host` với màu sắc phù hợp

### YAML/JSON

- Sử dụng 2 spaces cho indentation
- Đảm bảo valid syntax

## Testing

- Test thay đổi của bạn trước khi submit PR
- Đảm bảo không có lỗi syntax
- Kiểm tra các services hoạt động đúng

## Documentation

- Cập nhật README.md nếu thêm tính năng mới
- Cập nhật CHANGELOG.md với thay đổi của bạn
- Thêm comments trong code khi cần

## Questions?

Nếu có câu hỏi, hãy tạo issue với label `question`.

Cảm ơn bạn đã đóng góp! 🙏

