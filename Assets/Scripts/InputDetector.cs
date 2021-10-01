using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using OpenCvSharp;

namespace Visual3DInput{
    public class InputDetector : MonoBehaviour
    {
        public WebCamTexture cam_texture;
        public bool displayable = false;

        private CascadeClassifier classifier;
        private int last_position;
        private bool ready = false;

        public void Awake(){ 
            WebCamDevice[] device_list = WebCamTexture.devices;
            cam_texture = new WebCamTexture(device_list[0].name);
            cam_texture.Play();
            classifier = new CascadeClassifier(Application.dataPath + @"/haarcascade_frontalface_default.xml");
        }

        public void Update()
        {
            Debug.Log(getInputDirection());
        }

        private OpenCvSharp.Rect findFace(Mat frame)
        {
            OpenCvSharp.Rect[] faces = classifier.DetectMultiScale(frame, 1.1, 2, HaarDetectionType.ScaleImage);
            if(faces.Length > 0)
            {
                if(!ready){
                    ready = true;
                    last_position = faces[0].X + (faces[0].Width / 2);
                }
                if(displayable)
                {
                    GetComponent<Renderer>().material.mainTexture = cam_texture;
                    frame.Rectangle(faces[0], new Scalar(0, 255, 0), 5);
                    GetComponent<Renderer>().material.mainTexture = OpenCvSharp.Unity.MatToTexture(frame);
                }
                return faces[0];
            }
            return new OpenCvSharp.Rect();
        }

        public float getInputDirection(){
            Mat frame = OpenCvSharp.Unity.TextureToMat(cam_texture);
            OpenCvSharp.Rect current_face = findFace(frame);
            int xPosition = current_face.X + (current_face.Width / 2);
            float normalizedDirection = Mathf.Clamp(xPosition - last_position, -1, 1);
            last_position = xPosition;
            return normalizedDirection;
        }
    }
}
