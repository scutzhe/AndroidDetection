package com.facesdk;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.android.gms.common.api.GoogleApiClient;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

import static android.content.ContentValues.TAG;

public class MainActivity extends Activity {
    private static final int SELECT_IMAGE = 1;

    private TextView infoResult;
    private ImageView imageView;
    private Bitmap yourSelectedImage = null;

    private FaceSDKNative faceSDKNative = new FaceSDKNative();
    /**
     * ATTENTION: This was auto-generated to implement the App Indexing API.
     * See https://g.co/AppIndexing/AndroidStudio for more information.
     */
    private GoogleApiClient client;

    //Check Permissions
    private static final int REQUEST_CODE_PERMISSION = 2;
    private static String[] PERMISSIONS_REQ = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.CAMERA
    };

    @Override
    protected void onDestroy() {
        super.onDestroy();
        faceSDKNative.FaceDetectionModelUnInit();
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        // For API 23+ you need to request the read/write permissions even if they are already in your manifest.
        int currentapiVersion = Build.VERSION.SDK_INT;
        if (currentapiVersion >= Build.VERSION_CODES.M) {
            verifyPermissions(this);
        }

        //copy model
        try {
            copyBigDataToSD("nme_min_aug_96_cpu.mnn");
        } catch (IOException e) {
            e.printStackTrace();
        }

        // 获取外部文件夹
        File sdDir = Environment.getExternalStorageDirectory();//get model store dir
        // facesdk's dir store model'file
        String sdPath = sdDir.toString() + "/facesdk/";
        faceSDKNative.FaceDetectionModelInit(sdPath);

         // store result'file
//        String txtDir = sdDir.toString() + "/results/";
//        File fileTxtDir = null;
//        try{
//            fileTxtDir = new File(txtDir);
//            if(!fileTxtDir.exists()){
//                fileTxtDir.mkdir();
//            }
//        } catch (Exception e){
//            e.printStackTrace();
//        }
//        Log.i(TAG,"results:"+txtDir);
//        String filePath = txtDir + "/" + "result_arm82.txt";
//        File fileTxt = null;
//        try{
//            fileTxt = new File(filePath);
//            if(!fileTxt.exists()){
//                fileTxt.createNewFile();
//            }
//        }catch (Exception e){
//            e.printStackTrace();
//        }
//        FileWriter writer = null;

        // do one test
        String tmpImagePath = "/storage/emulated/0/test_face/01.jpg";
        try{
            FileInputStream tmpInput = new FileInputStream(tmpImagePath);
            Bitmap tmpBitmap = BitmapFactory.decodeStream(tmpInput);
            int tmpWidth = tmpBitmap.getWidth();
            int tmpHeight = tmpBitmap.getHeight();
            byte[] tmpImageData = getPixelsRGBA(tmpBitmap);
            float faceKeyPoint[] =  faceSDKNative.FaceDetection(tmpImageData, tmpWidth, tmpHeight,4);
//            for(int i=0;i<faceKeyPoint.length;i++){
//                Log.i(TAG,"feature="+faceKeyPoint[i]);
//                writer = new FileWriter(fileTxt, true);
//                writer.append(Float.toString(faceKeyPoint[i]) + "\r\n");
//                writer.flush();
//            }
        }catch (FileNotFoundException e) {
            e.printStackTrace();
        }
//        catch (IOException e) {
//            e.printStackTrace();
//        }

        ////do batch test
        String imageDir = sdDir.toString() + "/test_face/";
        File file = new File(imageDir);
        File[] names = file.listFiles();
        String imagePath = "";
        int index = 0;
        long startTime = System.currentTimeMillis();
        for (File name : names) {
            index += 1;
            imagePath = "" + name;
//            Log.i(TAG, "imagePath:"+imagePath);
            try {
                FileInputStream input = new FileInputStream(imagePath);
                Bitmap bitmap = BitmapFactory.decodeStream(input);
                int width = bitmap.getWidth();
                int height = bitmap.getHeight();
//                Log.i(TAG,"width,height:"+width+","+height);
                byte[] imageData = getPixelsRGBA(bitmap);
                // do FaceDetection
                float faceKeyPoint[] = faceSDKNative.FaceDetection(imageData, width, height, 4);
//                Log.i(TAG,"keyPoint_size:"+faceKeyPoint.length);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
        long timeTotal = System.currentTimeMillis() - startTime;
        Log.i(TAG,"index:"+index);
        Log.i(TAG,"timeCost:"+timeTotal/index);

          ////do single test
//        infoResult = (TextView) findViewById(R.id.infoResult);
//        imageView = (ImageView) findViewById(R.id.imageView);
//        Button buttonImage = (Button) findViewById(R.id.buttonImage);
//        buttonImage.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View arg0) {
//                Intent i = new Intent(Intent.ACTION_PICK);
//                i.setType("image/*");
//                startActivityForResult(i, SELECT_IMAGE);
//            }
//        });
//        Button buttonDetect = (Button) findViewById(R.id.buttonDetect);
//        buttonDetect.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View arg0) {
//                if (yourSelectedImage == null)
//                    return;
//                int width = yourSelectedImage.getWidth();
//                int height = yourSelectedImage.getHeight();
//                Log.i(TAG,"width,height:"+width+","+height);
//                byte[] imageDate = getPixelsRGBA(yourSelectedImage);
//
//                long timeDetectFace = System.currentTimeMillis();
//                //do FaceDetection
//                Log.i(TAG, " start detection ...");
//                float faceKeyPoint[] =  faceSDKNative.FaceDetection(imageDate, width, height,4);
//                Log.i(TAG,"keyPoint_size:"+faceKeyPoint.length);
//                timeDetectFace = System.currentTimeMillis() - timeDetectFace;
//
//                //Get Results
//               infoResult.setText("time cost："+timeDetectFace+"ms");
//               Log.i(TAG, "time cost："+timeDetectFace);
//               Bitmap drawBitmap = yourSelectedImage.copy(Bitmap.Config.ARGB_8888, true);
//               Log.i(TAG,"length:"+faceKeyPoint.length);
//               for (int i=0; i<98; i++) {
//                   Canvas canvas = new Canvas(drawBitmap);
//                   Paint paint = new Paint();
//                   paint.setColor(Color.RED);
//                   paint.setStyle(Paint.Style.STROKE);
//                   paint.setStrokeWidth(5);
//                   //Draw rect
//                   float x = faceKeyPoint[i*2] * 96;
//                   float y = faceKeyPoint[i*2 + 1] * 96;
//                   Log.i(TAG,"x,y:"+x+","+y);
//                   canvas.drawCircle(x,y,0.1f, paint);
//               }
//                Log.i(TAG,"yaw,pitch,roll:"+faceKeyPoint[196] * 180 /Math.PI +","
//                        +faceKeyPoint[197] * 180 / Math.PI+","
//                        +faceKeyPoint[198]* 180 / Math.PI);
//
//                imageView.setImageBitmap(drawBitmap);
//            }
//        });
//        // ATTENTION: This was auto-generated to implement the App Indexing API.
//        // See https://g.co/AppIndexing/AndroidStudio for more information.
//        client = new GoogleApiClient.Builder(this).addApi(AppIndex.API).build();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();

            try {
                if (requestCode == SELECT_IMAGE) {
                    Bitmap bitmap = decodeUri(selectedImage);
                    Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                    yourSelectedImage = rgba;
                    imageView.setImageBitmap(yourSelectedImage);
                }
            } catch (FileNotFoundException e) {
                Log.e("MainActivity", "FileNotFoundException");
                return;
            }
        }
    }

    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException {
        // Decode image size
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

        // The new size we want to scale to
        final int REQUIRED_SIZE = 400;

        //// Find the correct scale value. It should be the power of 2.
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE
                    || height_tmp / 2 < REQUIRED_SIZE) {
                break;
            }
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        //// Decode with inSampleSize
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        return BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o2);
    }

    private byte[] getPixelsRGBA(Bitmap image) {
        // calculate how many bytes our image consists of
        int bytes = image.getByteCount();
        ByteBuffer buffer = ByteBuffer.allocate(bytes); // Create a new buffer
        image.copyPixelsToBuffer(buffer); // Move the byte data to the buffer
        byte[] temp = buffer.array(); // Get the underlying array containing the

        return temp;
    }

    private void copyBigDataToSD(String strOutFileName) throws IOException {
        File sdDir = Environment.getExternalStorageDirectory();//get root dir
        File file = new File(sdDir.toString()+"/facesdk/");
        if (!file.exists()) {
            file.mkdir();
        }

        String tmpFile = sdDir.toString()+"/facesdk/" + strOutFileName;
        File f = new File(tmpFile);
        if (f.exists()) {
            Log.i(TAG, "file exists " + strOutFileName);
            return;
        }
        InputStream myInput;
        java.io.OutputStream myOutput = new FileOutputStream(sdDir.toString()+"/facesdk/"+ strOutFileName);
        myInput = this.getAssets().open(strOutFileName);
        byte[] buffer = new byte[1024];
        int length = myInput.read(buffer);
        while (length > 0) {
            myOutput.write(buffer, 0, length);
            length = myInput.read(buffer);
        }
        myOutput.flush();
        myInput.close();
        myOutput.close();
        Log.i(TAG, "end copy file " + strOutFileName);

    }

    /**
     * Checks if the app has permission to write to device storage or open camera
     * If the app does not has permission then the user will be prompted to grant permissions
     *
     * @param activity
     */
    private static boolean verifyPermissions(Activity activity) {
        // Check if we have write permission
        int write_permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        int read_permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.READ_EXTERNAL_STORAGE);
        int camera_permission = ActivityCompat.checkSelfPermission(activity, Manifest.permission.CAMERA);

        if (write_permission != PackageManager.PERMISSION_GRANTED ||
                read_permission != PackageManager.PERMISSION_GRANTED ||
                camera_permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                    activity,
                    PERMISSIONS_REQ,
                    REQUEST_CODE_PERMISSION
            );
            return false;
        } else {
            return true;
        }
    }

}
