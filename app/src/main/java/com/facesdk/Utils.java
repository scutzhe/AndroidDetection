package com.facesdk;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class Utils {
    public void createFile(String txtDir,String fileName) throws IOException {
        File dir = new File(txtDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        File checkFile = new File(txtDir + "/" +fileName);
        FileWriter writer = null;
        try {
            if (!checkFile.exists()) {
                checkFile.createNewFile();
            }
            // 向目标文件中写入内容
            // FileWriter(File file, boolean append)，append为true时为追加模式，false或缺省则为覆盖模式
            writer = new FileWriter(checkFile, true);
            writer.append("your content" + "\r\n");
            writer.flush();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (null != writer)
                writer.close();
        }
    }
}
