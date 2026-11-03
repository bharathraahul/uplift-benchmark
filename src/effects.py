import numpy as np


def smd(df, feature):
    """SMD of a feature between the ad and no-ad groups."""
    ad = df[df['treatment']==1][feature]
    no_ad = df[df['treatment']==0][feature]
    return (ad.mean() - no_ad.mean()) / np.sqrt((ad.var() + no_ad.var()) / 2)


def ate(df, outcome):
    """Return the ad group mean, no-ad group mean, ATE (their difference) and its standard error."""
    ad = df[df['treatment']==1][outcome]
    no_ad = df[df['treatment']==0][outcome]
    se = np.sqrt(ad.var()/len(ad) + no_ad.var()/len(no_ad))
    return ad.mean(), no_ad.mean(), ad.mean() - no_ad.mean(), se
