"""Kokoro worker process and its wire protocol (self.speak#5).

P1 lands ``codec`` alone: the framed streaming format, with no worker and no
behaviour change. The format is the risky part of moving Kokoro out of main, so
it gets its own phase and its own review.
"""
