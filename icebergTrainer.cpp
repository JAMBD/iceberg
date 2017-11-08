#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "ml.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <list>
#include <algorithm>
#include <iterator>
#include <jsoncpp/json/json.h>

using namespace cv;
using namespace ml;
using namespace std;

Mat tmp45map;
Mat tmp45msk;
Mat tmp90map;
Mat tmp90msk;
Mat tmp180map;
Mat tmp180msk;
bool maskGen = false;

class dataPnt{
    private:
        void genMask(Mat& map, Mat& msk, int ang){
            Mat flt = Mat::zeros(75*2,75*2,CV_32F);
            Mat rlt = Mat::zeros(75*2,75*2,CV_32F);
            for(int i=0;i<75;i+=1){
                for(int j=0;j<75;j+=1){
                    Mat back = Mat::zeros(75*3-1,75*3-1,CV_32F);
                    Mat tmp = Mat::zeros(75,75,CV_32F);
                    Mat center=back(Rect(75,75,75,75));
                    tmp.at<float>(Point(i,j)) = 1;
                    tmp.copyTo(center);
                    Mat r = getRotationMatrix2D(Point(75*1.5,75*1.5), ang, 1.0);
                    warpAffine(back, back, r, back.size());

                    Mat match = Mat::zeros(75*2,75*2,CV_32F);
                    matchTemplate( back, tmp, match, CV_TM_CCORR);

                    rlt += match * ((float)i*75+j);
                    flt += match;
                }
            }
            rlt.copyTo(map);
            flt.copyTo(msk);
        }
        void setupMask(){
            if(!maskGen){
                genMask(tmp45map,tmp45msk,135);
                genMask(tmp90map,tmp90msk,90);
                genMask(tmp180map,tmp180msk,180);
                maskGen = true;
            }
        }

        void angCorrFeature(Mat input, int ang, Point& center, double& score){
            angCorrFeature(input, input,ang,center,score);
        }

        void histf(Mat input, vector<float>& hist){
            int histmax;
            int histloc;
            histf(input,hist,histmax,histloc);
        }

        void histf(Mat input, vector<float>& hist,int& histmax,int& histloc){
            hist.clear();
            for(int i=0;i<256;++i)
                hist.push_back(0);
            double minVal; 
            double maxVal; 
            Point minLoc; 
            Point maxLoc;
            histmax = 0;
            histloc = 0;
            minMaxLoc(input, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
            for(int i=0;i<75;++i){
                for(int j=0;j<75;++j){
                    float normed = (input.at<float>(Point(i,j)) - minVal)/(maxVal-minVal);
                    int hcnt = hist[(int)(normed*255)] ++;
                    if(hcnt > histmax){
                        histmax = hcnt;
                        histloc = (int)(normed*255);
                    }
                }
            }
        }
        void angCorrFeature(Mat input, Mat mask, int ang, Point& center, double& score){
            Mat back = Mat::zeros(75*3-1,75*3-1,CV_32F);
            Mat cnr=back(Rect(75,75,75,75));
            input.copyTo(cnr);
            Mat r = getRotationMatrix2D(Point(75*1.5,75*1.5), ang, 1.0);
            warpAffine(back, back, r, back.size());

            Mat match = Mat::zeros(75*2,75*2,CV_32F);
            matchTemplate( back, mask, match, CV_TM_SQDIFF );
            Mat msk;
            Mat map;

            switch (ang){
                case 135:
                    {
                        tmp45msk.copyTo(msk);
                        tmp45map.copyTo(map);
                        break;
                    }
                case 90:
                    {
                        tmp90msk.copyTo(msk);
                        tmp90map.copyTo(map);
                        break;
                    }
                case 180:
                    {
                        tmp180msk.copyTo(msk);
                        tmp180map.copyTo(map);
                        break;
                    }
            }
            if(center.x > 0 && center.y > 0){
                int loc = center.x * 75 + center.y;
                map = abs(map - loc);
                double minVal; 
                double maxVal; 
                Point minLoc; 
                Point maxLoc;
                minMaxLoc(map, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
                score = match.at<float>(minLoc);
            }else{
                double minVal; 
                double maxVal; 
                Point minLoc; 
                Point maxLoc;
                //match = match.mul(msk);
                match = match + (1-msk);
                minMaxLoc(match, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
                int loc = map.at<float>(minLoc);
                int my = loc % 75;
                int mx = loc / 75;
                center = Point(mx,my);
                score = match.at<float>(minLoc);
            }
        }
    public:
        // from data
        float isIceberg;
        string id;
        float band1[75][75];
        float band2[75][75];
        float incAngle;
        bool hasAng;

        // generated
        int pixelCnt;
        float pixelSum;
        double scr45;
        double scr90;
        double scr180;
        double maxVal; 

        dataPnt(){
            setupMask();
        }

        void preprocess(){
            Mat img = Mat::zeros(75*2,75*2,CV_32F);
            Mat band1 = Mat::zeros(75,75,CV_32F);
            Mat band2 = Mat::zeros(75,75,CV_32F);
            Mat band1lin = Mat::zeros(75,75,CV_32F);
            Mat band2lin = Mat::zeros(75,75,CV_32F);
            Mat amp = Mat::zeros(75,75,CV_32F);
            Mat phi = Mat::zeros(75,75,CV_32F);
            float sum1 = 0;
            float sum2 = 0;
            for(int i=0;i<75;++i){
                for(int j=0;j<75;++j){
                    sum1 += this->band1[i][j];
                    sum2 += this->band2[i][j];
                }
            }

            sum1 /= (75*75);
            sum2 /= (75*75);
            float max1 = 0;
            float max2 = 0;
            float min1 = 0;
            float min2 = 0;
            for(int i=0;i<75;++i){
                for(int j=0;j<75;++j){
                    float v1 = this->band1[i][j];
                    float v2 = this->band2[i][j];
                    max1 = max1 > v1 ? max1 : v1;
                    max2 = max2 > v2 ? max2 : v2;
                    min1 = min1 < v1 ? min1 : v1;
                    min2 = min2 < v2 ? min2 : v2;
                    band1.at<float>(Point(i,j)) = (this->band1[i][j] - sum1)/50 + 0.0;
                    band2.at<float>(Point(i,j)) = (this->band2[i][j] - sum1)/50 + 0.0;
                    float val1 = pow(10.0f,(this->band1[i][j]/20));
                    float val2 = pow(10.0f,(this->band2[i][j]/20));
                    band1lin.at<float>(Point(i,j)) = val1;
                    band2lin.at<float>(Point(i,j)) = val2;
                    amp.at<float>(Point(i,j)) =  sqrt(val1*val1 + val2*val2);
                    phi.at<float>(Point(i,j)) = atan2(val2,val1)/CV_PI * 2;
                }
            }

            static int tcmp = 0;
            double minVal; 
            Point minLoc; 
            Point maxLoc;
            minMaxLoc(amp, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
            vector<float> hist;
            vector<float> histNormed;
            int histmax;
            int histloc;
            histf(amp,hist,histmax,histloc);
            for(int i=histloc;i<256;++i){
                if(hist[i]< histmax*0.2){
                    histloc = i;
                    break;
                }
            }

            Mat ampNormed;
            amp.copyTo(ampNormed);
            float bellpeak = (((float)histloc/256.0)*(maxVal-minVal)+minVal);
            ampNormed = ampNormed - bellpeak;
            ampNormed = ampNormed /  (maxVal - bellpeak);
            threshold(ampNormed,ampNormed,0,1,THRESH_TOZERO);
            histf(ampNormed,histNormed);
            //cout << bellpeak << endl;

            Mat tmp = img(Rect(0,0,75,75));
            ampNormed.copyTo(tmp);
            Mat histimg = Mat::zeros(512,512,CV_8UC3);
            pixelCnt = 0;
            for(int i=0;i<256;++i){
                pixelCnt += histNormed[i];
                line(histimg,Point(0,i*2),Point(histNormed[i],i*2),Scalar(0,255,0));
                line(histimg,Point(0,i*2+1),Point(hist[i],i*2+1),Scalar(255,0,0));
            }
            pixelCnt -= histNormed[0];
            pixelSum = sum(ampNormed)[0];
            //cout << pixelCnt << "," << pixelSum << endl;
            //imshow("hist",histimg);
            if(this->hasAng){
                float ang = this->incAngle/180.0*CV_PI;
                float len = 20;
                //line(tmp,Point(len*sin(ang),len*cos(ang))+maxLoc,Point(-len*sin(ang),-len*cos(ang))+maxLoc,Scalar(1));
            }
            angCorrFeature(ampNormed,ampNormed,135,maxLoc,scr45);
            angCorrFeature(ampNormed,ampNormed,90,maxLoc,scr90);
            angCorrFeature(ampNormed,ampNormed,180,maxLoc,scr180);
            //cout << scr45 << "," << scr90 << "," << scr180<< endl;
            circle(tmp,maxLoc,7,Scalar(0),1);
            circle(tmp,maxLoc,6,Scalar(1),1);

            tmp = img(Rect(75,0,75,75));
            band1lin.copyTo(tmp);

            tmp = img(Rect(0,75,75,75));
            phi.copyTo(tmp);

            tmp = img(Rect(75,75,75,75));
            band2lin.copyTo(tmp);

            resize(img,img, cvSize(0, 0), 5, 5,INTER_NEAREST );
            //imshow("img",img);
            //waitKey(1);
        }
};

class classifer{
    public:
        virtual void reset()=0;
        virtual void train(vector<dataPnt> input) = 0;
        virtual void classify(vector<dataPnt>& input) = 0;
};

class randomClassifer:public classifer{
    public:
        void reset(){

        }
        void train(vector<dataPnt> input){
            std::srand(std::time(0));
        }
        void classify(vector<dataPnt>& input){
            for(auto & i:input){
                i.isIceberg = ((float)rand()/RAND_MAX);
            }
        }
};

class simpleClassifer:public classifer{
    private:
        Mat graph;
        int posCutoff1;
        int posCutoff2;
    public:
        simpleClassifer(){
        }
        void reset(){
            graph = Mat::zeros(800,600,CV_8UC3);
            posCutoff1 = 0;
            posCutoff2 = 0;
        }
        void train(vector<dataPnt> input){
            for(auto k:input){
                line(graph,Point(k.scr180*5,k.scr45*5),Point(k.scr90*5, k.scr45*5),Scalar(k.isIceberg>0.5?0:255,k.isIceberg>0.5?255:0,0),1);
                circle(graph,Point(k.pixelCnt/2, k.pixelSum*3),1,Scalar(k.isIceberg>0.5?0:255,k.isIceberg>0.5?255:0,0),-1);
            }
            imshow("graph",graph);
        }
        void classify(vector<dataPnt>& input){
            for(auto & i:input){
                i.isIceberg = ((float)rand()/RAND_MAX);
            }
        }
};

class neuralClassifer:public classifer{
    private:
        Mat layers;
        Ptr<ANN_MLP> nnetwork;
        int input_size;
    public:
        neuralClassifer(){
            input_size = 6;
            Mat_<int> layers(4,1);
            layers(0) = input_size;     // input
            layers(1) = 15;  // hidden
            layers(2) = 10;  // hidden
            layers(3) = 1;      // output, 1 pin per class.
            nnetwork = ANN_MLP::create();
            nnetwork->setLayerSizes(layers);
            nnetwork->setActivationFunction( ANN_MLP::SIGMOID_SYM, 0.6, 1);
            nnetwork->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 3000, 0.0001));
            nnetwork->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.0001);
        }
        void reset(){
        }
        void train(vector<dataPnt> input){
            Mat training_data = Mat(input.size(),input_size,CV_32FC1);
            Mat training_output = Mat(input.size(),1,CV_32FC1);
            for(int i=0;i<input.size();++i){
                training_data.at<float>(Point(0,i)) = input[i].pixelCnt;
                training_data.at<float>(Point(1,i)) = input[i].pixelSum;
                training_data.at<float>(Point(2,i)) = input[i].scr45;
                training_data.at<float>(Point(3,i)) = input[i].scr90;
                training_data.at<float>(Point(4,i)) = input[i].scr180;
                training_data.at<float>(Point(5,i)) = input[i].maxVal; 
                training_output.at<float>(Point(0,i)) = 2*(input[i].isIceberg-0.5); 
            }
            int iterations = nnetwork->train(training_data, ROW_SAMPLE,training_output);
            cout << iterations << endl;
        }
        void classify(vector<dataPnt>& input){
            Mat test_data = Mat(input.size(),input_size,CV_32FC1);
            Mat test_output = Mat(input.size(),1,CV_32FC1);
            for(int i=0;i<input.size();++i){
                test_data.at<float>(Point(0,i)) = input[i].pixelCnt;
                test_data.at<float>(Point(1,i)) = input[i].pixelSum;
                test_data.at<float>(Point(2,i)) = input[i].scr45;
                test_data.at<float>(Point(3,i)) = input[i].scr90;
                test_data.at<float>(Point(4,i)) = input[i].scr180;
                test_data.at<float>(Point(5,i)) = input[i].maxVal; 
            }
            nnetwork->predict(test_data,test_output);
            for(int i=0;i<input.size();++i){
                input[i].isIceberg = (test_output.at<float>(Point(0,i)) + 1)/2.0;
            }
        }
};

class tester{
    vector<dataPnt> fullset;

    public:
    int splits;
    tester(vector<dataPnt> inputSet){
        fullset = inputSet;
        splits = 20;
    }
    void test(classifer* dut){
        vector<dataPnt> shuffled = fullset;
        random_shuffle ( shuffled.begin(), shuffled.end());
        vector<vector<dataPnt> > splitSet(splits);
        for(int i=0;i<fullset.size();++i){
            splitSet[i%splits].push_back(fullset[i]);
        }
        float tt = 0;
        float tf = 0;
        float ft = 0;
        float ff = 0;
        for(int i=0;i<splits;++i){
            float ltt = 0;
            float ltf = 0;
            float lft = 0;
            float lff = 0;
            dut->reset();
            vector<bool> truth;
            vector<dataPnt> toTrain;
            vector<dataPnt> toTest;
            toTest = splitSet[i];
            for(int j=0;j<splits;++j){
                if(j!=i){
                    toTrain.insert(toTrain.end(),splitSet[j].begin(),splitSet[j].end());
                }
            }
            random_shuffle ( toTrain.begin(), toTrain.end());
            for(auto k:toTest){
                truth.push_back(k.isIceberg);
            }
            dut->train(toTrain);
            dut->classify(toTest);
            for(int j=0;j<truth.size();++j){
                if(toTest[j].isIceberg > 0.5){
                    if(truth[j]){
                        ltt = ltt + 1;
                    }else{
                        ltf = ltf + 1;
                    }
                }else{
                    if(truth[j]){
                        lft = lft + 1;
                    }else{
                        lff = lff + 1;
                    }
                }
            }
            tt += ltt;
            tf += ltf;
            ft += lft;
            ff += lff;
            cout << (ltt/toTest.size()) << "," << (ltf/toTest.size()) << "," << (lft/toTest.size()) << "," << (lff/toTest.size()) << endl;
            cout << "Accuracy: " << ((ltt+lff)/toTest.size()) << endl;
        }
        tt /= fullset.size();
        tf /= fullset.size();
        ft /= fullset.size();
        ff /= fullset.size();
        cout << tt << "," << tf << "," << ft << "," << ff << endl;
        cout << "Accuracy: " << (tt+ff) << endl;
    }
};

int main(int argc, char** argv){
    srand(time(0));
    Json::Value root;
    Json::Reader reader;
    const Json::Value defValue; //used for default reference
    ifstream ifile("data/train.json");

    Json::Value root_test;
    Json::Reader reader_test;
    const Json::Value defValue_test; //used for default reference
    ifstream ifile_test("data/test.json");
    
    bool isJsonOK = (ifile != NULL && reader.parse(ifile, root));
    isJsonOK &= (ifile_test != NULL && reader_test.parse(ifile_test, root_test));

    classifer* dut = new neuralClassifer();

    if (isJsonOK) {
        vector<dataPnt> trainingData;
        for(int i=0; i<root.size(); ++i){
            dataPnt tmp;
            if(!root[i]["inc_angle"].isString()){
                tmp.hasAng = true;
                tmp.incAngle = root[i]["inc_angle"].asFloat();
            }else{
                tmp.hasAng = false;
            }
            tmp.id = root[i]["id"].asString();
            tmp.isIceberg = root[i]["is_iceberg"].asFloat();
            for(int j=0; j<75; ++j){
                for(int k=0;k<75;++k){
                    tmp.band1[j][k] =  root[i]["band_1"][j*75+k].asFloat();
                    tmp.band2[j][k] =  root[i]["band_2"][j*75+k].asFloat();
                }
            }
            tmp.preprocess();
            trainingData.push_back(tmp);
        }
        cout << "Training set size: " <<  trainingData.size() << endl;
        tester experiment(trainingData);
        dut->reset();
        experiment.test(dut);

        vector<dataPnt> testData;
        for(int i=0;i>root_test.size(); ++i){
            dataPnt tmp;
            if(!root_test[i]["inc_angle"].isString()){
                tmp.hasAng = true;
                tmp.incAngle = root_test[i]["inc_angle"].asFloat();
            }else{
                tmp.hasAng = false;
            }
            tmp.id = root_test[i]["id"].asString();
            tmp.isIceberg = root_test[i]["is_iceberg"].asFloat();
            for(int j=0; j<75; ++j){
                for(int k=0;k<75;++k){
                    tmp.band1[j][k] =  root_test[i]["band_1"][j*75+k].asFloat();
                    tmp.band2[j][k] =  root_test[i]["band_2"][j*75+k].asFloat();
                }
            }
            tmp.preprocess();
            testData.push_back(tmp);
        }
        dut->classify(testData);
        //waitKey();
    } else
        cout << "json is not correct format !!" << endl;
    return 0;
}
