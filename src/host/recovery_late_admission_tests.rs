// Exercise the real post-FIFO hook, Session/Generation and completion/release paths.
fn join_after_admission(cohort: &mut StartupCohort, sessions: &mut Vec<Session>,
    pool: &mut Option<PageReservations>, now: Instant) {
    cohort.after_admission(sessions, pool, &mut vec![], &mut vec![], now);
}

#[test]
fn late_join_is_disabled_without_both_policy_switches() {
    for (recovery, late) in [(false, false), (false, true), (true, false)] {
        let mut pool = Some(PageReservations::new(64,256,CONTEXT).unwrap());
        let (a,_ar)=session(pool.as_mut().unwrap(),128,4);
        let (b,_br)=session(pool.as_mut().unwrap(),512,4);
        let mut sessions=vec![a,b]; let first=cohort_work(&sessions[0]);
        let mut cohort=StartupCohort::with_late_admission(recovery,late);
        cohort.maybe_start(&mut sessions,&[first],Instant::now());
        let (c,_cr)=session(pool.as_mut().unwrap(),128,4); sessions.push(c);
        let before=cohort.snapshot_state().map(|(ids,s,d)|(ids.to_vec(),s,d)).unwrap();
        join_after_admission(&mut cohort,&mut sessions,&mut pool,Instant::now());
        let after=cohort.snapshot_state().map(|(ids,s,d)|(ids.to_vec(),s,d)).unwrap();
        assert_eq!(before,after);
        assert!(!sessions[2].delivery.held()&&!sessions[2].delivery.once_participated());
        assert!(!cohort.allows(&sessions[2]));
    }
}

#[test]
fn late_join_only_fresh_prompt_keeps_visible_and_invisible_mixed_decode_unheld() {
    let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
    let (old,old_rx)=session(pool.as_mut().unwrap(),128,4);
    let (silent,silent_rx)=stop_prefix_session(pool.as_mut().unwrap(),true);
    let (a,a_rx)=session(pool.as_mut().unwrap(),128,4);
    let (b,_br)=session(pool.as_mut().unwrap(),512,4);
    let mut sessions=vec![old,silent,a,b];
    let old_ring=sessions[0].blocks.sliding; let silent_ring=sessions[1].blocks.sliding;
    cohort_complete(&mut sessions,old_ring,3,&mut pool);
    cohort_complete(&mut sessions,silent_ring,3,&mut pool);
    assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Delta(t)) if t=="a"));
    assert!(matches!(silent_rx.try_recv(),Err(mpsc::TryRecvError::Empty)));
    let first=cohort_work(&sessions[2]);
    let mut cohort=StartupCohort::with_late_admission(true,true);
    cohort.maybe_start(&mut sessions,&[first],Instant::now());
    cohort_complete(&mut sessions,first.ring,3,&mut pool);
    let (late,late_rx)=session(pool.as_mut().unwrap(),128,4); sessions.push(late);
    let pages=pool.as_ref().unwrap().available_pages();
    let receipts:Vec<_>=sessions.iter().map(|s|s.timing.received).collect();
    join_after_admission(&mut cohort,&mut sessions,&mut pool,Instant::now());
    assert!(sessions[..2].iter().all(|s|cohort.allows(s)&&!s.delivery.held()&&!s.delivery.once_participated()));
    assert!(sessions[4].delivery.held()&&sessions[4].delivery.once_participated()&&cohort.allows(&sessions[4]));
    let work:Vec<_>=[0,1,4].map(|i|cohort_work(&sessions[i])).into();
    assert!(complete_work(&mut sessions,&work,Some(&[SampleOutput::Greedy(4),SampleOutput::Greedy(5),SampleOutput::Greedy(3)]),
        &mut pool,&mut vec![],&mut vec![]).unwrap());
    assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Delta(t)) if t=="b"));
    assert!(matches!(silent_rx.try_recv(),Ok(StreamEvent::Delta(t)) if t=="ac"));
    assert!(matches!(a_rx.try_recv(),Err(mpsc::TryRecvError::Empty)));
    assert!(matches!(late_rx.try_recv(),Err(mpsc::TryRecvError::Empty)));
    assert_eq!(sessions[4].generation.completion_tokens(),1);
    assert!(!cohort.allows(&sessions[4]),"no extra decode behind a held first output");
    assert!(sessions[4].timing.progress.first.is_none());
    assert_eq!(sessions.iter().map(|s|s.timing.received).collect::<Vec<_>>(),receipts);
    assert_eq!(pool.as_ref().unwrap().available_pages(),pages);
    assert!(complete_work(&mut sessions,&work,Some(&[SampleOutput::Greedy(4),SampleOutput::Greedy(5),SampleOutput::Greedy(3)]),
        &mut pool,&mut vec![],&mut vec![]).is_err());
    assert_eq!(sessions[0].generation.completion_tokens(),2);
    assert_eq!(cohort.release_reason(&sessions,Instant::now()),None);
}

#[test]
fn late_join_keeps_started_and_deadline_and_earlier_receipt_can_only_shrink() {
    let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
    let start=Instant::now();
    let (mut a,_ar)=session(pool.as_mut().unwrap(),128,4);
    let (mut b,_br)=session(pool.as_mut().unwrap(),512,4);
    a.timing.received=start+Duration::from_millis(1000); b.timing.received=a.timing.received;
    let mut sessions=vec![a,b]; let first=cohort_work(&sessions[0]);
    let mut cohort=StartupCohort::with_late_admission(true,true);
    let captured=start+Duration::from_millis(1500);
    cohort.maybe_start(&mut sessions,&[first],captured);
    let (mut c,_cr)=session(pool.as_mut().unwrap(),128,4); c.timing.received=start+Duration::from_millis(2000); sessions.push(c);
    join_after_admission(&mut cohort,&mut sessions,&mut pool,start+Duration::from_millis(2100));
    let original_deadline=start+Duration::from_millis(1000)+startup_cohort::RECEIPT_BUDGET;
    assert_eq!(cohort.snapshot_state().unwrap().1,captured);
    assert_eq!(cohort.snapshot_state().unwrap().2,original_deadline);
    assert_eq!(cohort.snapshot_state().unwrap().0.len(),3);
    join_after_admission(&mut cohort,&mut sessions,&mut pool,start+Duration::from_millis(2200));
    assert_eq!(cohort.snapshot_state().unwrap().0.len(),3,"no duplicate join");
    // Simulate a previously queued original receipt; admission does not renew it.
    let (mut older,_or)=session(pool.as_mut().unwrap(),512,4); older.timing.received=start; sessions.push(older);
    join_after_admission(&mut cohort,&mut sessions,&mut pool,start+Duration::from_millis(2300));
    let deadline=start+startup_cohort::RECEIPT_BUDGET;
    assert_eq!(cohort.snapshot_state().unwrap().1,captured);
    assert_eq!(cohort.snapshot_state().unwrap().2,deadline);
    assert_eq!(cohort.snapshot_state().unwrap().0.len(),4);
    assert_eq!(cohort.release_reason(&sessions,deadline-Duration::from_nanos(1)),None);
    assert_eq!(cohort.release_reason(&sessions,deadline),Some("deadline"));
    release_recovery(&mut cohort,&mut sessions,&mut pool,deadline,"deadline");
    assert!(sessions.iter().all(|s|s.delivery.once_participated()&&!s.delivery.held()&&cohort.allows(s)));
    cohort.maybe_start(&mut sessions,&[first],deadline);
    assert!(!cohort.active(),"late participants cannot be captured again");
}

#[test]
fn late_join_ready_or_host_time_expiration_releases_before_adding_arrivals() {
    for ready in [false,true] {
        let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap()); let start=Instant::now();
        let (mut a,ar)=session(pool.as_mut().unwrap(),128,1);
        let (mut b,br)=session(pool.as_mut().unwrap(),128,4);
        a.timing.received=start;b.timing.received=start;
        let mut sessions=vec![a,b];let first=cohort_work(&sessions[0]);let b_ring=sessions[1].blocks.sliding;
        let mut cohort=StartupCohort::with_late_admission(true,true);
        cohort.maybe_start(&mut sessions,&[first],start+Duration::from_millis(10));
        cohort_complete(&mut sessions,first.ring,3,&mut pool);
        if ready { cohort_complete(&mut sessions,b_ring,crate::host::generate::EOS_TOKEN,&mut pool); }
        let (mut late,_lr)=session(pool.as_mut().unwrap(),128,4);late.timing.id=100;late.timing.received=start+Duration::from_millis(100);
        sessions.push(late);
        let now=if ready {start+Duration::from_millis(200)} else {start+startup_cohort::RECEIPT_BUDGET};
        join_after_admission(&mut cohort,&mut sessions,&mut pool,now);
        assert!(!cohort.active());
        let late=sessions.iter().find(|s|s.timing.id==100).unwrap();
        assert!(!late.delivery.held()&&!late.delivery.once_participated()&&cohort.allows(late));
        assert!(matches!(ar.try_recv(),Ok(StreamEvent::Delta(t)) if t=="a"));
        assert!(matches!(ar.try_recv(),Ok(StreamEvent::Done(_))));
        assert!(matches!(ar.try_recv(),Err(mpsc::TryRecvError::Disconnected)));
        if ready {
            assert_eq!(sessions.len(),1);
            assert!(matches!(br.try_recv(),Ok(StreamEvent::Done(o)) if o.completion_tokens==0));
            assert_eq!(pool.as_ref().unwrap().available_pages(),255);
        } else {
            assert_eq!(sessions.len(),2);
            assert!(sessions.iter().find(|s|s.blocks.sliding==b_ring).unwrap().reservation.is_some());
            assert_eq!(pool.as_ref().unwrap().available_pages(),254);
        }
    }
}

#[test]
fn late_join_expired_queued_receipt_flushes_without_marking_or_skipping_it() {
    let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());let start=Instant::now();
    let (mut a,ar)=session(pool.as_mut().unwrap(),128,4);let (mut b,_br)=session(pool.as_mut().unwrap(),512,4);
    a.timing.received=start+Duration::from_millis(1000);b.timing.received=a.timing.received;
    let mut sessions=vec![a,b];let first=cohort_work(&sessions[0]);let mut cohort=StartupCohort::with_late_admission(true,true);
    cohort.maybe_start(&mut sessions,&[first],start+Duration::from_millis(1500));cohort_complete(&mut sessions,first.ring,3,&mut pool);
    let (mut expired,_er)=session(pool.as_mut().unwrap(),128,4);expired.timing.received=start;
    let (mut young,_yr)=session(pool.as_mut().unwrap(),128,4);young.timing.received=start+Duration::from_millis(3500);
    sessions.extend([expired,young]);let pages=pool.as_ref().unwrap().available_pages();
    let now=start+Duration::from_millis(4000);assert_eq!(cohort.release_reason(&sessions,now),None);
    join_after_admission(&mut cohort,&mut sessions,&mut pool,now);
    assert!(!cohort.active());assert!(matches!(ar.try_recv(),Ok(StreamEvent::Delta(t)) if t=="a"));
    assert!(sessions[2..].iter().all(|s|!s.delivery.held()&&!s.delivery.once_participated()));
    assert_eq!(pool.as_ref().unwrap().available_pages(),pages);
    let young_work=cohort_work(&sessions[3]);cohort.maybe_start(&mut sessions,&[young_work],now);
    assert!(!cohort.active(),"expired oldest cannot be skipped to rescue a younger subset");
}

#[test]
fn late_join_preserves_prior_checkpoint_after_old_decoder_completion_compacts_sessions() {
    let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());let start=Instant::now();
    let (old,old_rx)=session(pool.as_mut().unwrap(),128,2);
    let (mut a,ar)=session(pool.as_mut().unwrap(),128,1);let (mut b,br)=session(pool.as_mut().unwrap(),1024,4);
    a.timing.received=start;b.timing.received=start+Duration::from_millis(1000);
    let mut sessions=vec![old,a,b];let old_ring=sessions[0].blocks.sliding;let b_ring=sessions[2].blocks.sliding;
    let b_pages=sessions[2].blocks.full.clone();cohort_complete(&mut sessions,old_ring,3,&mut pool);
    let first=cohort_work(&sessions[1]);let mut cohort=StartupCohort::with_late_admission(true,true);
    cohort.maybe_start(&mut sessions,&[first],start+Duration::from_millis(2000));cohort_complete(&mut sessions,first.ring,3,&mut pool);
    release_recovery(&mut cohort,&mut sessions,&mut pool,start+startup_cohort::RECEIPT_BUDGET,"deadline");
    assert!(matches!(ar.try_recv(),Ok(StreamEvent::Delta(_))));assert!(matches!(ar.try_recv(),Ok(StreamEvent::Done(_))));
    let intermediate=ScheduledInput{ring:b_ring,position:0,count:512};
    assert!(complete_work(&mut sessions,&[intermediate],Some(&[SampleOutput::Logits(vec![])]),&mut pool,&mut vec![],&mut vec![]).unwrap());
    let checkpoint=ScheduledInput{ring:b_ring,position:512,count:512};
    let mut fresh_receivers=Vec::new();
    for id in [100,101] {
        let (mut fresh,rx)=session(pool.as_mut().unwrap(),128,4);fresh.timing.id=id;fresh.timing.received=start+Duration::from_millis(3000);
        // Keep channels alive while only testing their prompt eligibility.
        fresh_receivers.push(rx);sessions.push(fresh);
    }
    let fresh=cohort_work(sessions.iter().find(|s|s.timing.id==100).unwrap());
    cohort.maybe_start(&mut sessions,&[fresh],start+Duration::from_millis(3800));
    let (mut late,_lr)=session(pool.as_mut().unwrap(),512,4);late.timing.id=102;late.timing.received=start+Duration::from_millis(3900);sessions.push(late);
    join_after_admission(&mut cohort,&mut sessions,&mut pool,start+Duration::from_millis(4000));
    assert_eq!(cohort.snapshot_state().unwrap().0,&[100,101,102]);
    assert_eq!(sessions.iter().position(|s|s.blocks.sliding==b_ring),Some(1));
    cohort_complete(&mut sessions,old_ring,4,&mut pool);
    assert_eq!(sessions.iter().position(|s|s.blocks.sliding==b_ring),Some(0));
    assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Delta(t)) if t=="a"));assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Delta(t)) if t=="b"));assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Done(_))));
    let pages=pool.as_ref().unwrap().available_pages();drop(br);
    assert!(!complete_work(&mut sessions,&[checkpoint],None,&mut pool,&mut vec![],&mut vec![]).unwrap());
    let prior=&sessions[0];assert_eq!(prior.blocks.full,b_pages);assert_eq!(prior.generation.input().1,512);
    assert!(!prior.delivery.held()&&prior.delivery.once_participated()&&cohort.allows(prior));
    assert_eq!(pool.as_ref().unwrap().available_pages(),pages);
    let available=recovery_ready(&sessions,&cohort);let active=[b_ring];
    let selected=select_work_in(64,&available,&mut true,&mut 0,&active,true);
    assert_eq!(schedule_inputs(&available,&selected,&[checkpoint],true),vec![checkpoint]);
    assert!(complete_work(&mut sessions,&[checkpoint],Some(&[SampleOutput::Greedy(5)]),&mut pool,&mut vec![],&mut vec![]).unwrap());
    assert!(sessions.iter().all(|s|s.blocks.sliding!=b_ring));
    assert_eq!(pool.as_ref().unwrap().available_pages(),pages+b_pages.len());
    assert_eq!(cohort.snapshot_state().unwrap().0,&[100,101,102]);
}

#[test]
fn late_join_terminal_cancellation_and_fatal_flush_keep_exact_event_order() {
    let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
    let (a,ar)=session(pool.as_mut().unwrap(),128,1);let (b,br)=session(pool.as_mut().unwrap(),512,4);
    let mut sessions=vec![a,b];let first=cohort_work(&sessions[0]);let mut cohort=StartupCohort::with_late_admission(true,true);
    cohort.maybe_start(&mut sessions,&[first],Instant::now());cohort_complete(&mut sessions,first.ring,3,&mut pool);
    let (mut done,dr)=session(pool.as_mut().unwrap(),128,1);done.stream=false;let done_ring=done.blocks.sliding;
    let (eos,er)=session(pool.as_mut().unwrap(),128,4);let eos_ring=eos.blocks.sliding;
    let (cancel,cr)=session(pool.as_mut().unwrap(),128,4);let cancel_ring=cancel.blocks.sliding;
    sessions.extend([done,eos,cancel]);join_after_admission(&mut cohort,&mut sessions,&mut pool,Instant::now());
    drop(cr);drop(br);
    cohort_complete(&mut sessions,done_ring,4,&mut pool);
    cohort_complete(&mut sessions,eos_ring,crate::host::generate::EOS_TOKEN,&mut pool);
    cohort_complete(&mut sessions,cancel_ring,crate::axes::W::SIZE,&mut pool);
    assert_eq!(sessions.len(),5,"terminal held owners retain reservations");
    let pages=pool.as_ref().unwrap().available_pages();
    assert!(sessions.iter().filter(|s|s.delivery.terminal()).all(|s|!cohort.allows(s)));
    release_recovery(&mut cohort,&mut sessions,&mut pool,Instant::now(),"error");
    assert_eq!(sessions.len(),1);assert!(sessions[0].generation.is_prefilling()&&sessions[0].reservation.is_some());
    assert_eq!(pool.as_ref().unwrap().available_pages(),pages+4);
    for s in &mut sessions {let _=s.events.send(StreamEvent::Error("offline".into()));release_session(s,&mut pool,&mut vec![],&mut vec![]);}
    sessions.clear();assert_eq!(pool.as_ref().unwrap().available_pages(),256);
    assert!(matches!(ar.try_recv(),Ok(StreamEvent::Delta(t)) if t=="a"));assert!(matches!(ar.try_recv(),Ok(StreamEvent::Done(_))));
    assert!(matches!(dr.try_recv(),Ok(StreamEvent::Done(o)) if o.text=="b"&&o.completion_tokens==1));
    assert!(matches!(er.try_recv(),Ok(StreamEvent::Done(o)) if o.completion_tokens==0));
    for rx in [ar,dr,er] {assert!(matches!(rx.try_recv(),Err(mpsc::TryRecvError::Disconnected)),"no error/delta after terminal Done");}
    let(tx,rx)=mpsc::channel::<()>();drop(tx);assert!(idle_batch::receive(&rx,64).is_none());
}

#[test]
fn late_join_capacity_is_bounded_by_retained_reservations_and_pending_receipts_stay_old() {
    let mut pool=Some(PageReservations::new(4,16,CONTEXT).unwrap());let start=Instant::now();
    let (old,old_rx)=session(pool.as_mut().unwrap(),128,2);
    let (mut a,_ar)=session(pool.as_mut().unwrap(),128,1);let (mut b,_br)=session(pool.as_mut().unwrap(),512,4);
    a.timing.received=start;b.timing.received=start;
    let mut sessions=vec![old,a,b];let old_ring=sessions[0].blocks.sliding;cohort_complete(&mut sessions,old_ring,3,&mut pool);
    let first=cohort_work(&sessions[1]);let mut cohort=StartupCohort::with_late_admission(true,true);
    cohort.maybe_start(&mut sessions,&[first],start+Duration::from_millis(100));cohort_complete(&mut sessions,first.ring,3,&mut pool);
    let(mut c,_cr)=session(pool.as_mut().unwrap(),128,1);c.timing.received=start+Duration::from_millis(200);let c_ring=c.blocks.sliding;sessions.push(c);
    join_after_admission(&mut cohort,&mut sessions,&mut pool,start+Duration::from_millis(300));
    cohort_complete(&mut sessions,c_ring,3,&mut pool);
    let before=pool.as_ref().unwrap().available_pages();assert!(pool.as_mut().unwrap().try_reserve(516).unwrap().is_none());
    assert_eq!(pool.as_ref().unwrap().available_pages(),before);
    assert_eq!(cohort.snapshot_state().unwrap().0.len(),3);
    cohort_complete(&mut sessions,old_ring,4,&mut pool);
    assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Delta(_))));assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Delta(_))));assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Done(_))));
    let(mut d,_dr)=session(pool.as_mut().unwrap(),512,4);d.timing.id=100;d.timing.received=start+Duration::from_millis(250);d.timing.admitted=start+Duration::from_millis(900);sessions.push(d);
    join_after_admission(&mut cohort,&mut sessions,&mut pool,start+Duration::from_millis(900));
    assert_eq!(cohort.snapshot_state().unwrap().0.len(),4);
    assert_eq!(cohort.snapshot_state().unwrap().2,start+startup_cohort::RECEIPT_BUDGET);
    assert!(sessions.iter().all(|s|s.delivery.held()&&s.delivery.once_participated()));
    assert_eq!(sessions.iter().find(|s|s.timing.id==100).unwrap().timing.received,start+Duration::from_millis(250));
    assert!(pool.as_mut().unwrap().try_reserve(1).unwrap().is_none());
}

#[test]
fn late_join_release_compacts_below_a_live_prior_checkpoint_before_selection() {
    let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());let start=Instant::now();
    let(mut a,ar)=session(pool.as_mut().unwrap(),128,1);let(mut b,_br)=session(pool.as_mut().unwrap(),1024,4);
    a.timing.received=start;b.timing.received=start;
    let mut sessions=vec![a,b];let first=cohort_work(&sessions[0]);let b_ring=sessions[1].blocks.sliding;
    let b_pages=sessions[1].blocks.full.clone();let mut cohort=StartupCohort::with_late_admission(true,true);
    cohort.maybe_start(&mut sessions,&[first],start+Duration::from_millis(100));cohort_complete(&mut sessions,first.ring,3,&mut pool);
    let intermediate=ScheduledInput{ring:b_ring,position:0,count:512};
    assert!(complete_work(&mut sessions,&[intermediate],Some(&[SampleOutput::Logits(vec![])]),&mut pool,&mut vec![],&mut vec![]).unwrap());
    let checkpoint=ScheduledInput{ring:b_ring,position:512,count:512};
    assert!(!complete_work(&mut sessions,&[checkpoint],None,&mut pool,&mut vec![],&mut vec![]).unwrap());
    let(mut late,_lr)=session(pool.as_mut().unwrap(),128,4);late.timing.id=100;late.timing.received=start+Duration::from_millis(3700);sessions.push(late);
    let pages=pool.as_ref().unwrap().available_pages();assert_eq!(sessions[1].blocks.sliding,b_ring);
    join_after_admission(&mut cohort,&mut sessions,&mut pool,start+startup_cohort::RECEIPT_BUDGET);
    assert!(!cohort.active());assert_eq!(sessions[0].blocks.sliding,b_ring);
    assert_eq!(sessions[0].blocks.full,b_pages);assert_eq!(sessions[0].generation.input().1,512);
    assert!(sessions[0].delivery.once_participated()&&!sessions[0].delivery.held()&&cohort.allows(&sessions[0]));
    assert!(!sessions[1].delivery.once_participated()&&!sessions[1].delivery.held());
    assert_eq!(pool.as_ref().unwrap().available_pages(),pages+1);
    assert!(matches!(ar.try_recv(),Ok(StreamEvent::Delta(_))));assert!(matches!(ar.try_recv(),Ok(StreamEvent::Done(_))));
    let available=recovery_ready(&sessions,&cohort);
    let selected=select_work_in(64,&available,&mut true,&mut 0,&[b_ring],true);
    assert_eq!(schedule_inputs(&available,&selected,&[checkpoint],true),vec![checkpoint]);
}
