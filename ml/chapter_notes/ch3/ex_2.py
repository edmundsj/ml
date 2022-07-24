def shift(data, direction='up', padding=0):
    """
    Shifts the MNIST pixel data by an amount in each direction, padding
    """
    # How do we actually perform this shift? In the case of an 'up' shift, we need all
    # the values to shift up. The easiest way to do this is likely by re-indexing. 
    # However, we can't just shift all the pixels over by one, this would be snake-ily distorting the image
    if direction == 'up':
        data

