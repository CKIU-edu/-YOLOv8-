# YOLOv8 药片检测计数系统

<details open>
<summary><strong>中文（默认）</strong></summary>

## 项目介绍
轻量高效的药片检测与计数工具，基于 YOLOv8 实现实时视觉检测，核心功能包括：
- 加密模型管理：支持私有协议加密模型的导入、解密、切换
- 实时检测：摄像头实时识别药片并计数
- 数据留存：支持检测过程录像、当前帧截图
- 自定义配置：可设置缓存目录存放模型/配置文件

无需复杂部署，开箱即可用于医药场景的药片快速计数、批量核验等需求。


## 核心功能
1. **多模型管理**：导入/删除/重命名加密模型，支持多模型切换使用
2. **实时检测**：摄像头实时采集画面，自动识别药片并显示数量
3. **数据留存**：检测过程录像保存、当前帧截图导出
4. **安全加密**：私有协议加密模型，使用时自动解密（退出自动清理临时文件）
5. **自定义配置**：手动设置缓存目录，存储模型配置与临时文件


## 使用步骤
1. **设置缓存目录**：启动程序后，点击「系统设置」→「设置」选择缓存路径
2. **导入加密模型**：点击「导入模型」选择 `.rp` 格式加密模型，输入显示名称
3. **加载模型**：在模型列表选中目标模型，点击「使用选中」完成加载
4. **启动检测**：选择摄像头索引 → 点击「打开摄像头」→ 点击「开始检测」


## 环境依赖
- Python 3.8+
- 依赖库：`ultralytics opencv-python tkinter`


## 常见问题（FAQ）
1. **Q：导入模型时提示“解密失败”怎么办？**
   A：确认模型文件是 `.rp` 格式的加密文件，且与当前程序的加密协议匹配（密钥一致）。

2. **Q：摄像头无法打开怎么办？**
   A：检查摄像头索引是否正确（默认0，可尝试1/2），或关闭其他占用摄像头的程序。

3. **Q：检测计数不准确如何优化？**
   A：调整「置信度阈值」滑块（建议0.5-0.7），或更换更适配当前场景的模型。

4. **Q：临时文件会残留吗？**
   A：不会，程序退出时会自动清理缓存目录下的临时解密模型文件。

5. **Q：可以批量导入模型吗？**
   A：目前暂不支持批量导入，需逐个导入并设置显示名称。
</details>


<details>
<summary><strong>English</strong></summary>

## Project Introduction
A lightweight and efficient pill detection & counting tool based on YOLOv8, with core features:
- Encrypted Model Management: Import, decrypt, and switch models with private protocol encryption
- Real-time Detection: Camera-based real-time pill recognition and counting
- Data Retention: Record detection videos and export frame screenshots
- Custom Configuration: Set cache directory to store models and config files

Ready to use out of the box for rapid pill counting and batch verification in pharmaceutical scenarios.


## Core Features
1. **Multi-model Management**: Import/delete/rename encrypted models, support multi-model switching
2. **Real-time Detection**: Camera-based real-time frame capture, automatic pill recognition & counting
3. **Data Retention**: Save detection videos and export current frame screenshots
4. **Secure Encryption**: Private-protocol encrypted models, auto-decrypted when in use (temp files cleaned on exit)
5. **Custom Configuration**: Manually set cache directory for model configs and temp files


## Usage Steps
1. **Set Cache Directory**: After launching, go to "System Settings" → "Set" to choose cache path
2. **Import Encrypted Model**: Click "Import Model" to select `.rp` encrypted model, input display name
3. **Load Model**: Select target model in the list, click "Use Selected" to load
4. **Start Detection**: Choose camera index → Click "Open Camera" → Click "Start Detection"


## Dependencies
- Python 3.8+
- Libraries: `ultralytics opencv-python tkinter`
</details>


<details>
<summary><strong>日本語</strong></summary>

## プロジェクト紹介
YOLOv8 を基盤とした軽量・高効率な錠剤検出・計数ツールで、主な機能は以下の通りです：
- 暗号化モデル管理：プライベートプロトコルで暗号化されたモデルのインポート、復号、切り替えに対応
- リアルタイム検出：カメラでリアルタイムに錠剤を認識・計数
- データ保存：検出過程の録画、現在のフレームのスクリーンショットをエクスポート
- カスタム設定：キャッシュディレクトリを設定し、モデル・設定ファイルを保存

複雑なデプロイ不要で、医薬品シーンの錠剤の迅速な計数・一括検証などに即時利用可能です。


## コア機能
1. **マルチモデル管理**：暗号化モデルのインポート/削除/名前変更、マルチモデル切り替えに対応
2. **リアルタイム検出**：カメラでリアルタイムに画面を取得し、自動的に錠剤を認識・計数
3. **データ保存**：検出過程の録画保存、現在のフレームのスクリーンショットエクスポート
4. **安全な暗号化**：プライベートプロトコルで暗号化されたモデル、使用時に自動復号（終了時に一時ファイルを自動クリーンアップ）
5. **カスタム設定**：キャッシュディレクトリを手動設定し、モデル設定と一時ファイルを保存


## 使用手順
1. **キャッシュディレクトリの設定**：プログラム起動後、「システム設定」→「設定」でキャッシュパスを選択
2. **暗号化モデルのインポート**：「モデルをインポート」をクリックし、`.rp` 形式の暗号化モデルを選択し、表示名を入力
3. **モデルのロード**：モデルリストで対象モデルを選択し、「選択したものを使用」をクリックしてロード
4. **検出の開始**：カメラインデックスを選択 → 「カメラを開く」をクリック → 「検出を開始」をクリック


## 環境依存
- Python 3.8+
- 依存ライブラリ：`ultralytics opencv-python tkinter`
</details>


<details>
<summary><strong>한국어</strong></summary>

## 프로젝트 소개
YOLOv8을 기반으로 한 가벼우면서 효율적인 알약 검출 및 카운팅 도구로, 주요 기능은 다음과 같습니다：
- 암호화 모델 관리：프라이빗 프로토콜로 암호화된 모델의 임포트、복호、전환 지원
- 실시간 검출：카메라로 실시간으로 알약을 인식하고 카운팅
- 데이터 저장：검출 과정 녹화、현재 프레임 스크린샷 내보내기
- 사용자 정의 설정：캐시 디렉토리를 설정하여 모델・설정 파일 저장

복잡한 배포가 필요 없어 의약품 시나리오의 알약 신속 카운팅・일괄 검증 등에 즉시 사용 가능합니다.


## 핵심 기능
1. **멀티 모델 관리**：암호화 모델의 임포트/삭제/이름 변경、멀티 모델 전환 지원
2. **실시간 검출**：카메라로 실시간으로 화면을 획득하여 자동으로 알약 인식 및 카운팅
3. **데이터 저장**：검출 과정 녹화 저장、현재 프레임 스크린샷 내보내기
4. **안전한 암호화**：프라이빗 프로토콜로 암호화된 모델、사용 시 자동 복호（종료 시 임시 파일 자동 정리）
5. **사용자 정의 설정**：캐시 디렉토리를 수동 설정하여 모델 설정과 임시 파일 저장


## 사용 단계
1. **캐시 디렉토리 설정**：프로그램 실행 후 「시스템 설정」→「설정」으로 캐시 경로 선택
2. **암호화 모델 임포트**：「모델 임포트」를 클릭하여 `.rp` 형식의 암호화 모델을 선택하고 표시 이름 입력
3. **모델 로드**：모델 리스트에서 대상 모델을 선택하고 「선택 사용」을 클릭하여 로드
4. **검출 시작**：카메라 인덱스를 선택 → 「카메라 열기」 클릭 → 「검출 시작」 클릭


## 환경 의존
- Python 3.8+
- 의존 라이브러리：`ultralytics opencv-python tkinter`
</details>
