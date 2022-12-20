package com.example.seetrafficsignsopencv;

public enum ImageClass {
    SL_30(R.drawable.sl30),
    SL_40(R.drawable.sl40),
    SL_50(R.drawable.sl50),
    SL_60(R.drawable.sl60),
    SL_70(R.drawable.sl70),
    SL_80(R.drawable.sl80),
    SL_100(R.drawable.sl100),
    SL_120(R.drawable.sl120),
    EMPTY(R.drawable.sle);

    private final int id;

    ImageClass(int picID) {
        this.id = picID;
    }

    public int id() {
        return id;
    }
}
