import cv2
import numpy as np

def videoImprov(video_file, output_video, conversion_rate_list):
    '''
    입력 : 비디오 파일 경로, 저장할 비디오 이름, 화면전환율 리스트
    결과 : 화질개선된 비디오 파일 저장
    '''

    video = cv2.VideoCapture(video_file)

    if video.isOpened():
        run, img = video.read()
        h, w = img.shape[:2]

        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

        # 히스토그램 스트레칭을 위한 밝기 최대최소값 저장
        f_max = img.max()
        f_min = img.min()

        sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        # 영상이 끝날때까지 반복
        while True:
            run, img = video.read()

            i = 0
            if run:
                if conversion_rate_list[i] >= 0.55: # 화면전환율이 0.55이상일 경우 밝기 최대최소값 수정
                    f_max = img.max()
                    f_min = img.min()
                nframe = img.astype('int64')
                contrast_img = np.clip(((nframe - f_min) / (f_max - f_min)) * 255, 0, 255).astype('uint8')

                # 이미지 블러 처리
                contrast_img = contrast_img.astype(np.float32)
                blr_img = cv2.GaussianBlur(contrast_img, (0,0), 2)

                # 언샤프 마스크 생성 후 적용
                sharp_img = np.clip(2 * contrast_img - blr_img, 0, 255).astype(np.uint8)

                #sharpen = cv2.filter2D(sharp_img, -1, sharp_kernel)
                writer.write(sharp_img)

                i = i + 1

            else:
                break

        video.release()
        writer.release()
        cv2.destroyAllWindows()

    else:
        print('Can not open video..')
