import cv2
import numpy as np

def videoStab(video_file, output_video):
    '''
    입력 : 비디오 파일 경로, 저장할 비디오 이름
    결과 : 안정화된 비디오 파일 저장
    리턴 : 영상전체 프레임의 화면전환율 리스트
    '''

    # 변환할 영상 로드
    video = cv2.VideoCapture(video_file)

    if video.isOpened():
        run, img = video.read()
        h, w = img.shape[:2]

        img1 = img

        # 변환된 영상쓰기 객체 생성
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (int(w * 0.85), int(h * 0.85)))

        # 영상 내부 프레임 생성
        cut_frame = [int(h * 0.075), int(h * 0.925), int(w * 0.075), int(w * 0.925)]
        writer.write(img1[cut_frame[0]: cut_frame[1], cut_frame[2]: cut_frame[3]])

        # 화면전환율 리스트
        conversion_rate_list = []

        # 전후 프레임의 특징점 추출 및 매칭 객체 선언
        orb = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)

        # 영상이 끝날때까지 반복
        while True:
            run, img = video.read()

            if run:
                img2 = img

                # 검색시간을 줄이기 위한 검색영역 설정
                img1_box = img1[int(h / 4):int(3 * h / 4), int(w / 4):int(3 * w / 4)]
                img2_box = img2[int(h / 4):int(3 * h / 4), int(w / 4):int(3 * w / 4)]

                '''
                img1_box1 = img1[int(h / 6):int(2 * h / 6), int(w / 6):int(2 * w / 6)]
                img1_box2 = img1[int(h / 6):int(2 * h / 6), int(4 * w / 6):int(5 * w / 6)]
                img1_box3 = img1[int(4 * h / 6):int(5 * h / 6), int(w / 6):int(2 * w / 6)]
                img1_box4 = img1[int(4 * h / 6):int(5 * h / 6), int(4 * w / 6):int(5 * w / 6)]
                img1_up_box = cv2.hconcat([img1_box1,img1_box2])
                img1_down_box = cv2.hconcat([img1_box3,img1_box4])
                img1_box = cv2.vconcat([img1_up_box,img1_down_box])

                img2_box1 = img2[int(h / 6):int(2 * h / 6), int(w / 6):int(2 * w / 6)]
                img2_box2 = img2[int(h / 6):int(2 * h / 6), int(4 * w / 6):int(5 * w / 6)]
                img2_box3 = img2[int(4 * h / 6):int(5 * h / 6), int(w / 6):int(2 * w / 6)]
                img2_box4 = img2[int(4 * h / 6):int(5 * h / 6), int(4 * w / 6):int(5 * w / 6)]
                img2_up_box = cv2.hconcat([img2_box1,img2_box2])
                img2_down_box = cv2.hconcat([img2_box3,img2_box4])
                img2_box = cv2.vconcat([img2_up_box,img2_down_box])
                '''

                # 전후 프레임의 특징점 추출 및 매칭
                kp1, des1 = orb.detectAndCompute(img1_box, None)
                kp2, des2 = orb.detectAndCompute(img2_box, None)
                matches = matcher.knnMatch(des1, des2, 2)

                good = []
                pts1 = []
                pts2 = []
                for (m, n) in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                        pts1.append(kp1[m.queryIdx].pt)
                        pts2.append(kp2[m.trainIdx].pt)


                # 매칭결과 1개 미만인 경우는 화면전환으로 간주
                if len(good) <= 1:
                    conversion_rate_list.append(1)
                    writer.write(img2[cut_frame[0]: cut_frame[1], cut_frame[2]: cut_frame[3]])

                    img1 = img2
                    continue

                # 화면전환율
                conversion_rate = round(1 - len(good) / len(kp1), 2)
                conversion_rate_list.append(conversion_rate)

                # 내부 객체 움직임으로 인한 오차 줄이기
                raw_vectors = []
                for i in range(len(good)):
                    x = pts2[i][0] - pts1[i][0]
                    y = pts2[i][1] - pts1[i][1]
                    raw_vectors.append([x, -y])

                x_sum = 0
                y_sum = 0
                for i in raw_vectors:
                    x_sum = x_sum + i[0]
                    y_sum = y_sum + i[1]
                x_mean = round(x_sum / len(raw_vectors),1)
                y_mean = round(y_sum / len(raw_vectors),1)

                x_differ_sum = 0
                y_differ_sum = 0
                for i in raw_vectors:
                    x_differ_sum = x_differ_sum + abs(i[0] - x_mean)
                    y_differ_sum = y_differ_sum + abs(i[1] - y_mean)
                x_differ_mean = round(x_differ_sum / len(raw_vectors),1)
                y_differ_mean = round(y_differ_sum / len(raw_vectors),1)

                vectors = []
                for i in raw_vectors:
                    if abs(i[0] - x_mean) <= abs(x_differ_mean*1.4) and \
                            abs(i[1] - y_mean) <= abs(y_differ_mean*1.4):
                        vectors.append(i)

                # 최종 x,y 움직임 벡터 계산
                x_sum = 0
                y_sum = 0
                for i in vectors:
                    x_sum = x_sum + i[0]
                    y_sum = y_sum + i[1]
                final_x = int(x_sum / len(vectors))
                final_y = int(y_sum / len(vectors))

                # 내부프레임 이동 후 잘라내기
                if cut_frame[0]-final_y >= 0 and cut_frame[1]-final_y <= h and \
                        cut_frame[2] + final_x >= 0 and cut_frame[3] + final_x <= w:
                    cut_frame[0] = cut_frame[0] - final_y
                    cut_frame[1] = cut_frame[1] - final_y
                    cut_frame[2] = cut_frame[2] + final_x
                    cut_frame[3] = cut_frame[3] + final_x

                cut_img = img2[cut_frame[0]: cut_frame[1], \
                          cut_frame[2]: cut_frame[3]]
                writer.write(cut_img)

                img1 = img2

            else:
                break

        video.release()
        writer.release()
        cv2.destroyAllWindows()

    else:
        print('Can not open video..')
        conversion_rate_list = []

    return conversion_rate_list